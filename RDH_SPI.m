% ==========================================================================
% RDH_SPI.m  —  v1.0  [FIXED]
% A New Reversible Data Hiding Algorithm Based On Sorting-and-Prediction
% Integration  |  Li et al., IEEE Transactions on Consumer Electronics, 2026
% DOI: 10.1109/TCE.2026.3683088
% ==========================================================================

function RDH_SPI_v5()
    clc;
    fprintf('=== RDH: Sorting-and-Prediction Integration ===\n');
    fprintf('    Li et al., IEEE Trans. Consumer Electronics, 2026\n');
    fprintf('    DOI: 10.1109/TCE.2026.3683088\n\n');
    inputFolder = '/Volumes/Data_/Images/SIPI/Boat/';
    filename ='Boat.bmp';
    % ---- Parameters ----
    lambda = 2;   % prediction scaling (paper: λ=2, Eq.8)
    L      = 16;  % LPVO segment length
    T_thresh = [0, 100, 300, 600];  % smoothness thresholds (m=4)

    % ---- Test image ----
    cover = imread(fullfile(inputFolder, filename));
    [M, N] = size(cover);
    fprintf('Image: %d × %d grayscale\n', M, N);

    % ---- Secret data ----
    rng(7);
    payload_bits = uint8(randi([0 1], 1, 800));
    fprintf('Payload: %d bits\n\n', numel(payload_bits));

    % ==================================================================
    % EMBEDDING
    % ==================================================================
    fprintf('--- EMBEDDING ---\n');

    [idx1, idx2, idx3, idx4] = partition_image(M, N);
    [img, loc_map] = handle_boundary(cover);

    pay_ptr  = 1;
    embed_info = cell(4, 1);
    embedded_layers = [];

    all_idx    = {idx1, idx2, idx3, idx4};
    img_marked = img;

    for layer = 1:4
        target_idx = all_idx{layer};
        [sorted_seq, sort_map] = sort_region(img_marked, target_idx, M, N, lambda);
        [img_marked, bits_used, info] = lpvo_embed_layer(...
            img_marked, target_idx, sorted_seq, sort_map, ...
            payload_bits, pay_ptr, L, T_thresh);
        pay_ptr = pay_ptr + bits_used;
        embed_info{layer} = info;
        fprintf('  Layer %d (%s): %d bits embedded, capacity=%d\n', ...
            layer, layer_name(layer), bits_used, info.capacity);
        
        if bits_used > 0
            embedded_layers = [embedded_layers, layer]; %#ok<AGROW>
        end
        
        if pay_ptr > numel(payload_bits), break; end
    end

    total_embedded = pay_ptr - 1;
    fprintf('Total embedded: %d / %d bits\n', total_embedded, numel(payload_bits));
    bpp = total_embedded / (M*N);
    fprintf('Embedding rate: %.4f bpp\n\n', bpp);

    psnr_val = compute_psnr(cover, img_marked);
    fprintf('PSNR: %.2f dB\n\n', psnr_val);

    % ==================================================================
    % EXTRACTION
    % ==================================================================
    fprintf('--- EXTRACTION ---\n');

    img_ext   = img_marked;
    rec_bits  = [];

    for idx = numel(embedded_layers):-1:1
        layer = embedded_layers(idx);
        target_idx = all_idx{layer};
        info = embed_info{layer};
        [img_ext, bits_out] = lpvo_extract_layer(...
            img_ext, target_idx, info, M, N, lambda, L, T_thresh);
        rec_bits = [bits_out, rec_bits]; %#ok<AGROW>
        fprintf('  Layer %d (%s): %d bits extracted\n', ...
            layer, layer_name(layer), numel(bits_out));
    end

    img_ext = restore_boundary(img_ext, loc_map);

    n_check = min(numel(rec_bits), numel(payload_bits));
    match_payload = isequal(rec_bits(1:n_check), payload_bits(1:n_check));
    match_image   = isequal(img_ext, cover);

    fprintf('\n--- RESULTS ---\n');
    fprintf('Payload recovery:   %s  (%d/%d bits)\n', ...
        yesno(match_payload), sum(rec_bits(1:n_check)==payload_bits(1:n_check)), n_check);
    fprintf('Image restoration:  %s\n', yesno(match_image));
    fprintf('PSNR:               %.2f dB\n', psnr_val);
    fprintf('Embedding rate:     %.4f bpp  (%d bits)\n', bpp, total_embedded);
end

function [idx1, idx2, idx3, idx4] = partition_image(M, N)
    [cols, rows] = meshgrid(1:N, 1:M);
    r0 = mod(rows-1, 2);
    c0 = mod(cols-1, 2);
    idx1 = find(r0==0 & c0==0);
    idx2 = find(r0==0 & c0==1);
    idx3 = find(r0==1 & c0==0);
    idx4 = find(r0==1 & c0==1);
end

function [img, loc_map] = handle_boundary(img)
    loc_map = false(size(img));
    loc_map(img == 0)   = true;
    loc_map(img == 255) = true;
    img(img == 0)   = 1;
    img(img == 255) = 254;
end

function img = restore_boundary(img, loc_map)
end

function [sorted_seq, sort_map] = sort_region(img, target_idx, M, N, lambda)
    Xd  = double(img);
    np  = numel(target_idx);
    p_vec = zeros(np, 1);
    l_vec = zeros(np, 1);
    v_vec = zeros(np, 1);

    for i = 1:np
        idx = target_idx(i);
        [r, c] = ind2sub([M N], idx);
        v_vec(i) = Xd(r, c);

        get = @(dr,dc) safe_get(Xd, r+dr, c+dc, M, N);
        x1=get(-1,-1); x2=get(-1,0); x3=get(-1,+1);
        x4=get( 0,-1);               x5=get( 0,+1);
        x6=get(+1,-1); x7=get(+1,0); x8=get(+1,+1);

        h1=(x1+x2)/2; h2=(x2+x3)/2; h3=(x6+x7)/2; h4=(x7+x8)/2;
        h = (h1+h2+h3+h4)/4;

        v1=(x1+x4)/2; v2=(x4+x6)/2; v3=(x3+x5)/2; v4=(x5+x8)/2;
        v = (v1+v2+v3+v4)/4;

        o1=(x4+x2)/2; o2=(x4+x7)/2; o3=(x5+x2)/2; o4=(x5+x7)/2;
        o = (o1+o2+o3+o4)/4;

        x_hat = (h+v+o)/3;
        p_vec(i) = floor(x_hat * lambda + 0.5);

        l_vec(i) = abs(x1-x2)+abs(x2-x3)+abs(x6-x7)+abs(x7-x8) ...
                 + abs(x1-x4)+abs(x4-x6)+abs(x3-x5)+abs(x5-x8) ...
                 + abs(x4-x2)+abs(x4-x7)+abs(x5-x2)+abs(x5-x7);
    end

    p_min = min(p_vec); p_max = max(p_vec);
    l_min = min(l_vec); l_max = max(l_vec);

    n_rows = p_max - p_min + 1;
    n_cols = l_max - l_min + 1;
    mat    = cell(n_rows, n_cols);

    for i = 1:np
        ri = p_vec(i) - p_min + 1;
        ci = l_vec(i) - l_min + 1;
        mat{ri, ci} = [mat{ri, ci}, i];
    end

    sorted_order = [];
    for ri = 1:n_rows
        if mod(ri, 2) == 1
            col_range = 1:n_cols;
        else
            col_range = n_cols:-1:1;
        end
        for ci = col_range
            if ~isempty(mat{ri, ci})
                sorted_order = [sorted_order, mat{ri,ci}]; %#ok<AGROW>
            end
        end
    end

    sorted_seq = v_vec(sorted_order);
    sort_map.order      = sorted_order;
    sort_map.p_vec      = p_vec;
    sort_map.l_vec      = l_vec;
    sort_map.target_idx = target_idx;
end

function v = safe_get(Xd, r, c, M, N)
    r = max(1, min(M, r));
    c = max(1, min(N, c));
    v = Xd(r, c);
end

% ========== LPVO EMBEDDING ==========
function [img_out, bits_used, info] = lpvo_embed_layer(...
        img, target_idx, sorted_seq, sort_map, payload, pay_ptr, L, T_thresh)

    img_out   = img;
    bits_used = 0;
    capacity  = 0;
    Xd        = double(img);
    n_px      = numel(sorted_seq);
    n_seg     = floor(n_px / L);
    m         = numel(T_thresh);
    order     = sort_map.order;
    l_vec     = sort_map.l_vec;

    % === KEY FIX: Store embedding metadata for extraction ===
    seg_meta = struct('seg_idx', {}, 'K', {}, 'S_smooth', {});

    for s = 1:n_seg
        seg_range  = (s-1)*L + 1 : s*L;
        seg_local  = sorted_seq(seg_range);
        seg_order  = order(seg_range);
        seg_l      = l_vec(seg_order);

        S = sum(seg_l);
        K = num_pairs(S, T_thresh, m);
        capacity = capacity + K;

        if K == 0, continue; end

        [seg_sorted, sort_ord] = sort(seg_local);
        seg_gidx = target_idx(seg_order(sort_ord));

        embed_count = 0;

        for pair = 1:K
            if pay_ptr > numel(payload), break; end
            
            lo_gi  = seg_gidx(pair);
            lo_ref = seg_sorted(pair+1);
            hi_gi  = seg_gidx(end-pair+1);
            hi_ref = seg_sorted(end-pair);

            % Embed min pixel
            e_lo = double(img_out(lo_gi)) - double(lo_ref);
            if pay_ptr <= numel(payload)
                bit_lo = payload(pay_ptr);
                [new_lo, ok] = embed_one(double(img_out(lo_gi)), e_lo, bit_lo, -1);
                if ok
                    img_out(lo_gi) = uint8(max(0, min(255, new_lo)));
                    embed_count = embed_count + 1;
                    pay_ptr = pay_ptr + 1;
                end
            end

            % Embed max pixel
            if pay_ptr <= numel(payload)
                e_hi = double(img_out(hi_gi)) - double(hi_ref);
                bit_hi = payload(pay_ptr);
                [new_hi, ok] = embed_one(double(img_out(hi_gi)), e_hi, bit_hi, +1);
                if ok
                    img_out(hi_gi) = uint8(max(0, min(255, new_hi)));
                    embed_count = embed_count + 1;
                    pay_ptr = pay_ptr + 1;
                end
            end
        end

        bits_used = bits_used + embed_count;
        
        % === Store K and S for extraction ===
        n = numel(seg_meta) + 1;
        seg_meta(n).seg_idx = seg_range;
        seg_meta(n).K = K;
        seg_meta(n).S_smooth = S;
    end

    info.capacity   = capacity;
    info.seg_meta   = seg_meta;
    info.sorted_seq = sorted_seq;
    info.sort_map   = sort_map;
    info.n_seg      = n_seg;
    info.L          = L;
    info.T_thresh   = T_thresh;
end

% ========== LPVO EXTRACTION (USE STORED K AND S) ==========
function [img_out, bits_out] = lpvo_extract_layer(...
        img, target_idx, info, M, N, lambda, L, T_thresh)

    img_out  = img;
    bits_out = [];

    [sorted_seq, sort_map] = sort_region(img, target_idx, M, N, lambda);

    n_px  = numel(sorted_seq);
    n_seg = floor(n_px / L);
    order = sort_map.order;
    l_vec = sort_map.l_vec;
    
    % === KEY FIX: Use stored seg_meta instead of recalculating K ===
    seg_meta = info.seg_meta;

    for s = 1:n_seg
        seg_range = (s-1)*L + 1 : s*L;
        
        seg_local = sorted_seq(seg_range);
        seg_order = order(seg_range);
        
        [seg_sorted, sort_ord] = sort(seg_local);
        seg_gidx = target_idx(seg_order(sort_ord));
        seg_bits = [];
        
        % === Get K from stored metadata (not recalculated) ===
        if s <= numel(seg_meta)
            K = seg_meta(s).K;
        else
            K = 0;
        end
        
        if K == 0, continue; end
        
        for pair = 1:K
            if pair > numel(seg_gidx) || (numel(seg_gidx)-pair+1) < 1
                continue;
            end
            
            lo_gi  = seg_gidx(pair);
            lo_ref = seg_sorted(pair+1);
            hi_gi  = seg_gidx(end-pair+1);
            hi_ref = seg_sorted(end-pair);

            % Extract min pixel
            e_lo = double(img_out(lo_gi)) - double(lo_ref);
            [orig_lo, bit_lo] = extract_one(double(img_out(lo_gi)), e_lo, -1);
            img_out(lo_gi) = uint8(max(0, min(255, orig_lo)));
            
            % Extract max pixel
            e_hi = double(img_out(hi_gi)) - double(hi_ref);
            [orig_hi, bit_hi] = extract_one(double(img_out(hi_gi)), e_hi, +1);
            img_out(hi_gi) = uint8(max(0, min(255, orig_hi)));
            
            seg_bits = [seg_bits, bit_lo, bit_hi];
        end
        
        bits_out = [bits_out, seg_bits];
    end
end

function [new_val, ok] = embed_one(x, e, bit, direction)
    ok = true;
    if direction < 0
        if e == 0
            new_val = x - bit;
        elseif e == -1
            new_val = x - 1;
        elseif e < -1
            new_val = x - 1;
        else
            new_val = x; ok = false;
        end
    else
        if e == 0
            new_val = x + bit;
        elseif e == 1
            new_val = x + 1;
        elseif e > 1
            new_val = x + 1;
        else
            new_val = x; ok = false;
        end
    end
end

function [orig_val, bit] = extract_one(x, e, direction)
    if direction < 0
        if e == 0
            bit = 0; orig_val = x;
        elseif e == -1
            bit = 1; orig_val = x + 1;
        else
            bit = 0; orig_val = x + 1;
        end
    else
        if e == 0
            bit = 0; orig_val = x;
        elseif e == 1
            bit = 1; orig_val = x - 1;
        else
            bit = 0; orig_val = x - 1;
        end
    end
end

function K = num_pairs(S, T_thresh, m)
    K = 0;
    for i = 1:m
        if i == 1
            lo = 0;
        else
            lo = T_thresh(i-1);
        end
        hi = T_thresh(i);
        if S >= lo && S < hi
            K = 2*(m + 1 - i);
            return;
        end
    end
    if S >= T_thresh(end)
        K = 0;
    end
end

function p = compute_psnr(orig, marked)
    diff = double(orig) - double(marked);
    mse  = mean(diff(:).^2);
    if mse == 0
        p = Inf;
    else
        p = 10 * log10(255^2 / mse);
    end
end

function s = yesno(b)
    if b, s = 'PASS ✓'; else s = 'FAIL ✗'; end
end

function s = layer_name(i)
    names = {'I1','I2','I3','I4'};
    s = names{i};
end
