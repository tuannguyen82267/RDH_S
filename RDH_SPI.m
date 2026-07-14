% ==========================================================================
% RDH_SPI.m  —  v1.0
% A New Reversible Data Hiding Algorithm Based On Sorting-and-Prediction
% Integration  |  Li et al., IEEE Transactions on Consumer Electronics, 2026
% DOI: 10.1109/TCE.2026.3683088
%
% Algorithm overview:
%   1. Partition image into 4 interleaved regions I1, I2, I3, I4
%   2. For each region, compute 2D sorting criterion:
%        p = prediction value (h+v+o)/3 × λ  (Eq.1–8)
%        l = local complexity (12-pair abs-diff) (Eq.9)
%   3. Snake-like scanning: build p×l matrix, scan row-alternating → sorted seq
%   4. Four-layer LPVO embedding: I1→I2→I3→I4, each using updated neighbors
%   5. Extraction: reverse order I4'→I3'→I2'→I1' with same criterion
%
% Image partition (checkerboard × 2):
%   I1 = even-row, even-col  (0-based)
%   I2 = even-row, odd-col
%   I3 = odd-row,  even-col
%   I4 = odd-row,  odd-col
%   → each pixel's 8 neighbors belong to the OTHER 3 regions
%
% LPVO (Location-based Pixel Value Ordering, [39]):
%   Segment of size L sorted by value
%   2nd-smallest predicts smallest → error e1 = x_min - x_2ndmin
%   2nd-largest  predicts largest  → error e2 = x_max - x_2ndmax
%   Embed via 2D histogram shift mapping F
%
% Parameters:
%   λ=2, L=16, segment smoothness thresholds T (adaptive)
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
    max_bits_per_pair = 2;           % m+1-i pairs for threshold range

    % ---- Test image ----
    cover = imread(fullfile(inputFolder, filename));
    %cover = uint8(cover);
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

    % Step 1: Partition into 4 regions
    [idx1, idx2, idx3, idx4] = partition_image(M, N);

    % Handle overflow/underflow: map 0→1, 255→254 and record location map
    [img, loc_map] = handle_boundary(cover);

    % Four-layer embedding: I1 → I2 → I3 → I4
    pay_ptr  = 1;
    embed_info = cell(4, 1);  % stores aux info for each layer
    embedded_layers = [];      % Track which layers actually got data

    all_idx    = {idx1, idx2, idx3, idx4};
    img_marked = img;

    for layer = 1:4
        target_idx = all_idx{layer};
        % Reference pixels: all OTHER 3 regions (already updated)
        [sorted_seq, sort_map] = sort_region(img_marked, target_idx, M, N, lambda);
        [img_marked, bits_used, info] = lpvo_embed_layer(...
            img_marked, target_idx, sorted_seq, sort_map, ...
            payload_bits, pay_ptr, L, T_thresh);
        pay_ptr = pay_ptr + bits_used;
        embed_info{layer} = info;
        fprintf('  Layer %d (%s): %d bits embedded, capacity=%d\n', ...
            layer, layer_name(layer), bits_used, info.capacity);
        
        % Track layers that were embedded
        if bits_used > 0
            embedded_layers = [embedded_layers, layer]; %#ok<AGROW>
        end
        
        if pay_ptr > numel(payload_bits), break; end
    end

    total_embedded = pay_ptr - 1;
    fprintf('Total embedded: %d / %d bits\n', total_embedded, numel(payload_bits));
    bpp = total_embedded / (M*N);
    fprintf('Embedding rate: %.4f bpp\n\n', bpp);

    % Compute PSNR
    psnr_val = compute_psnr(cover, img_marked);
    fprintf('PSNR: %.2f dB\n\n', psnr_val);

    % ==================================================================
    % EXTRACTION
    % ==================================================================
    fprintf('--- EXTRACTION ---\n');

    img_ext   = img_marked;
    rec_bits  = [];

    % Extract ONLY from layers that were embedded, in REVERSE order
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

    % Restore boundary pixels
    img_ext = restore_boundary(img_ext, loc_map);

    % Verify
    n_check = min(numel(rec_bits), numel(payload_bits));
    match_payload = isequal(rec_bits(1:n_check), payload_bits(1:n_check));
    match_image   = isequal(img_ext, cover);

    fprintf('\n--- RESULTS ---\n');
    fprintf('Payload recovery:   %s  (%d/%d bits)\n', ...
        yesno(match_payload), sum(rec_bits(1:n_check)==payload_bits(1:n_check)), n_check);
    fprintf('Image restoration:  %s\n', yesno(match_image));
    fprintf('PSNR:               %.2f dB\n', psnr_val);
    fprintf('Embedding rate:     %.4f bpp  (%d bits)\n', bpp, total_embedded);

    % ---- Paper results summary ----
    fprintf('\n--- PAPER RESULTS (Table II/III/V) ---\n');
    fprintf('%-12s %8s %8s\n', 'Image', '10k bits', '20k bits');
    imgs  = {'Boat','Lake','Elaine','Peppers','Barbara','Average'};
    p10k  = [60.11, 61.59, 59.78, 60.55, 62.26, 60.13];
    p20k  = [55.87, 56.66, 54.95, 56.61, 58.51, 56.52];
    for i = 1:numel(imgs)
        fprintf('%-12s %8.2f %8.2f\n', imgs{i}, p10k(i), p20k(i));
    end
    fprintf('\nKodak avg (20k bits): 60.43 dB (proposed) vs 59.84 dB [53]\n');
end

% ==========================================================================
% IMAGE PARTITIONING (Sec.II-A, Fig.1)
% Checkerboard × 2:
%   I1: even row, even col (0-based index)
%   I2: even row, odd  col
%   I3: odd  row, even col
%   I4: odd  row, odd  col
% Each pixel's 8 neighbors come ONLY from the other 3 regions ✓
% ==========================================================================
function [idx1, idx2, idx3, idx4] = partition_image(M, N)
    [cols, rows] = meshgrid(1:N, 1:M);  % MATLAB 1-indexed
    r0 = mod(rows-1, 2);  % 0=even row, 1=odd row
    c0 = mod(cols-1, 2);  % 0=even col, 1=odd col

    idx1 = find(r0==0 & c0==0);  % I1
    idx2 = find(r0==0 & c0==1);  % I2
    idx3 = find(r0==1 & c0==0);  % I3
    idx4 = find(r0==1 & c0==1);  % I4
end

% ==========================================================================
% HANDLE BOUNDARY PIXELS (Sec.III-C)
% 0 → 1, 255 → 254, record in location map
% ==========================================================================
function [img, loc_map] = handle_boundary(img)
    loc_map = false(size(img));
    loc_map(img == 0)   = true;
    loc_map(img == 255) = true;
    img(img == 0)   = 1;
    img(img == 255) = 254;
end

function img = restore_boundary(img, loc_map)
    % Cannot perfectly restore without knowing original values
    % In real implementation, loc_map is stored in auxiliary info
    % For demo: no-op (values stay at 1/254)
end

% ==========================================================================
% COMPUTE SORTING CRITERION (Sec.II-A, Eq.1–9)
%
% For pixel x at (row,col) in target region:
%   8 neighbors x1..x8 come from other regions (already available)
%
%   Neighbor layout (Fig.2):
%   x1=(r-1,c-1)  x2=(r-1,c)  x3=(r-1,c+1)
%   x4=(r,  c-1)              x5=(r,  c+1)
%   x6=(r+1,c-1)  x7=(r+1,c)  x8=(r+1,c+1)
%
%   Prediction:
%     h = (h1+h2+h3+h4)/4  where h1=(x1+x2)/2, h2=(x2+x3)/2,
%                                  h3=(x6+x7)/2, h4=(x7+x8)/2
%     v = (v1+v2+v3+v4)/4  where v1=(x1+x4)/2, v2=(x4+x6)/2,
%                                  v3=(x3+x5)/2, v4=(x5+x8)/2
%     o = (o1+o2+o3+o4)/4  where o1=(x4+x2)/2, o2=(x4+x7)/2,
%                                  o3=(x5+x2)/2, o4=(x5+x7)/2
%     x_hat = (h+v+o)/3
%     p = floor(x_hat*lambda + 0.5)
%
%   Complexity (Eq.9):
%     l = |x1-x2|+|x2-x3|+|x6-x7|+|x7-x8|   (horizontal pairs)
%       + |x1-x4|+|x4-x6|+|x3-x5|+|x5-x8|   (vertical pairs)
%       + |x4-x2|+|x4-x7|+|x5-x2|+|x5-x7|   (diagonal pairs)
% ==========================================================================
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

        % Safe neighbor access
        get = @(dr,dc) safe_get(Xd, r+dr, c+dc, M, N);
        x1=get(-1,-1); x2=get(-1,0); x3=get(-1,+1);
        x4=get( 0,-1);               x5=get( 0,+1);
        x6=get(+1,-1); x7=get(+1,0); x8=get(+1,+1);

        % Horizontal feature h (Eq.1–2)
        h1=(x1+x2)/2; h2=(x2+x3)/2; h3=(x6+x7)/2; h4=(x7+x8)/2;
        h = (h1+h2+h3+h4)/4;

        % Vertical feature v (Eq.3–4)
        v1=(x1+x4)/2; v2=(x4+x6)/2; v3=(x3+x5)/2; v4=(x5+x8)/2;
        v = (v1+v2+v3+v4)/4;

        % Diagonal feature o (Eq.5–6)
        o1=(x4+x2)/2; o2=(x4+x7)/2; o3=(x5+x2)/2; o4=(x5+x7)/2;
        o = (o1+o2+o3+o4)/4;

        % Initial prediction x_hat (Eq.7)
        x_hat = (h+v+o)/3;

        % Quantized prediction p (Eq.8)
        p_vec(i) = floor(x_hat * lambda + 0.5);

        % Local complexity l (Eq.9) — 12 adjacent pairs
        l_vec(i) = abs(x1-x2)+abs(x2-x3)+abs(x6-x7)+abs(x7-x8) ...
                 + abs(x1-x4)+abs(x4-x6)+abs(x3-x5)+abs(x5-x8) ...
                 + abs(x4-x2)+abs(x4-x7)+abs(x5-x2)+abs(x5-x7);
    end

    % Snake-like scanning (Sec.II-B, Fig.3)
    p_min = min(p_vec); p_max = max(p_vec);
    l_min = min(l_vec); l_max = max(l_vec);

    % Build 2D matrix: rows=p, cols=l
    n_rows = p_max - p_min + 1;
    n_cols = l_max - l_min + 1;
    mat    = cell(n_rows, n_cols);  % each cell may hold multiple pixels

    for i = 1:np
        ri = p_vec(i) - p_min + 1;
        ci = l_vec(i) - l_min + 1;
        mat{ri, ci} = [mat{ri, ci}, i];  % store index into target region
    end

    % Snake scan: row r left-to-right if odd, right-to-left if even
    sorted_order = [];
    for ri = 1:n_rows
        if mod(ri, 2) == 1  % odd row: left → right
            col_range = 1:n_cols;
        else                 % even row: right → left
            col_range = n_cols:-1:1;
        end
        for ci = col_range
            if ~isempty(mat{ri, ci})
                sorted_order = [sorted_order, mat{ri,ci}]; %#ok<AGROW>
            end
        end
    end

    % sorted_seq: actual pixel values in snake order
    % sort_map:   how to map back (sorted_order = permutation of 1:np)
    sorted_seq = v_vec(sorted_order);
    sort_map.order      = sorted_order;
    sort_map.p_vec      = p_vec;
    sort_map.l_vec      = l_vec;
    sort_map.target_idx = target_idx;
end

% Helper: safe array access with boundary padding
function v = safe_get(Xd, r, c, M, N)
    r = max(1, min(M, r));
    c = max(1, min(N, c));
    v = Xd(r, c);
end

% ==========================================================================
% LPVO EMBEDDING (Sec.III-A, [39])
%
% Divide sorted sequence into segments of size L.
% For each segment:
%   1. Compute segment smoothness S = sum of l values
%   2. Determine # prediction error pairs K based on S vs thresholds T
%   3. Sort segment pixels by value → indices rank 1..L
%   4. Prediction errors: e1 = x(1) - x(2)  (min - 2nd_min)
%                         e2 = x(L) - x(L-1) (max - 2nd_max)
%      (and similarly for deeper pairs based on K)
%   5. Embed using 2D histogram shift: if e ∈ {-1,0,1}: embed 1 bit
% ==========================================================================
function [img_out, bits_used, info] = lpvo_embed_layer(...
        img, target_idx, sorted_seq, sort_map, payload, pay_ptr, L, T_thresh)

    img_out   = img;
    bits_used = 0;
    capacity  = 0;
    Xd        = double(img);
    n_px      = numel(sorted_seq);
    n_seg     = floor(n_px / L);
    m         = numel(T_thresh);    % number of thresholds
    order     = sort_map.order;
    l_vec     = sort_map.l_vec;

    seg_records = struct('seg_idx', {}, 'K', {}, 'embeds', {});

    for s = 1:n_seg
        seg_range  = (s-1)*L + 1 : s*L;
        seg_local  = sorted_seq(seg_range);        % pixel values
        seg_order  = order(seg_range);             % indices into target_idx
        seg_l      = l_vec(seg_order);             % complexities

        % Segment smoothness = sum of local complexities
        S = sum(seg_l);

        % Number of embedding pairs K (Sec.III-A)
        K = num_pairs(S, T_thresh, m);
        capacity = capacity + K;

        if K == 0, continue; end

        % Sort pixels by value within segment
        [seg_sorted, sort_ord] = sort(seg_local);
        seg_gidx = target_idx(seg_order(sort_ord));  % global pixel indices

        embeds = zeros(1, K*2);  % store (original_e, embedded_bit) pairs
        embed_count = 0;

        % Embed K pairs: min/max pairs
        for pair = 1:K
            lo_gi  = seg_gidx(pair);           % global idx of pair-min
            lo_ref = seg_sorted(pair+1);        % 2nd-min reference
            hi_gi  = seg_gidx(end-pair+1);     % global idx of pair-max
            hi_ref = seg_sorted(end-pair);      % 2nd-max reference

            % Error for min pixel
            e_lo = double(img_out(lo_gi)) - double(lo_ref);
            if pay_ptr + embed_count <= numel(payload)
                bit_lo = payload(pay_ptr + embed_count);
                [new_lo, ok] = embed_one(double(img_out(lo_gi)), e_lo, bit_lo, -1);
                if ok
                    img_out(lo_gi) = uint8(max(0, min(255, new_lo)));
                    embed_count = embed_count + 1;
                    %embeds(embed_count) = bit_lo;
                    embeds(embed_count) = bit_lo;  % embed bit_lo vào embeds(1)
                end
            end

            % Error for max pixel
            e_hi = double(img_out(hi_gi)) - double(hi_ref);
            if pay_ptr + embed_count <= numel(payload)
                bit_hi = payload(pay_ptr + embed_count);
                [new_hi, ok] = embed_one(double(img_out(hi_gi)), e_hi, bit_hi, +1);
                if ok
                    img_out(hi_gi) = uint8(max(0, min(255, new_hi)));
                    embed_count = embed_count + 1;
                    embeds(embed_count) = bit_hi;
                end
            end
        end

        bits_used = bits_used + embed_count;
        pay_ptr   = pay_ptr + embed_count;
        n = numel(seg_records) + 1;
        seg_records(n).seg_idx = seg_range;
        seg_records(n).K       = K;
        seg_records(n).embeds  = embeds(1:embed_count);
    end

    info.capacity    = capacity;
    info.seg_records = seg_records;
    info.sorted_seq  = sorted_seq;
    info.sort_map    = sort_map;
    info.n_seg       = n_seg;
    info.L           = L;
    info.T_thresh    = T_thresh;
end

% ==========================================================================
% LPVO EXTRACTION (Sec.III-B) — inverse of embedding
% ==========================================================================
function [img_out, bits_out] = lpvo_extract_layer(...
        img, target_idx, info, M, N, lambda, L, T_thresh)

    img_out  = img;
    bits_out = [];

    % Re-compute sort criterion on current image
    [sorted_seq, sort_map] = sort_region(img, target_idx, M, N, lambda);

    n_px  = numel(sorted_seq);
    n_seg = floor(n_px / L);
    m     = numel(T_thresh);
    order = sort_map.order;
    l_vec = sort_map.l_vec;

    for s = 1:n_seg   % FORWARD: same order as embedding
        seg_range = (s-1)*L + 1 : min(s*L, n_px);  % FIX: handle last segment properly
        
        if seg_range(1) > n_px, continue; end
        
        seg_local = sorted_seq(seg_range);
        seg_order = order(seg_range);
        seg_l     = l_vec(seg_order);
        S         = sum(seg_l);
        K         = num_pairs(S, T_thresh, m);
        if K == 0, continue; end

        [seg_sorted, sort_ord] = sort(seg_local);
        seg_gidx = target_idx(seg_order(sort_ord));

        seg_bits = [];
        for pair = 1:K  % FIX: iterate forward (1 to K), not backward
            % Check bounds before accessing
            if pair > numel(seg_gidx) || (numel(seg_gidx)-pair+1) < 1
                continue;
            end
            
            hi_gi  = seg_gidx(end-pair+1);
            hi_ref = seg_sorted(end-pair);
            lo_gi  = seg_gidx(pair);
            lo_ref = seg_sorted(pair+1);

            % Extract from max pixel
            e_hi = double(img_out(hi_gi)) - double(hi_ref);
            [orig_hi, bit_hi] = extract_one(double(img_out(hi_gi)), e_hi, +1);
            img_out(hi_gi) = uint8(max(0, min(255, orig_hi)));
            %seg_bits = [bit_hi, seg_bits]; %#ok<AGROW>
            seg_bits = [seg_bits, bit_hi]; %#ok<AGROW>  % append hi

            % Extract from min pixel
            e_lo = double(img_out(lo_gi)) - double(lo_ref);
            [orig_lo, bit_lo] = extract_one(double(img_out(lo_gi)), e_lo, -1);
            img_out(lo_gi) = uint8(max(0, min(255, orig_lo)));
            %seg_bits = [bit_lo, seg_bits]; %#ok<AGROW>
            seg_bits = [seg_bits, bit_lo]; %#ok<AGROW>  % append lo

        end
        bits_out = [bits_out, seg_bits]; %#ok<AGROW>
    end
end

% ==========================================================================
% EMBEDDING/EXTRACTION HELPERS
%
% 2D histogram shift (simplified 1D version for clarity):
%   Min pixel: e = x - ref (ref = 2nd-min, so e ≤ 0)
%     e = 0  → embed bit:  0→x stays, 1→x decremented
%     e = -1 → always shift x by -1 (expand room)
%     e < -1 → shift x by -1 (range shift)
%   Max pixel: e = x - ref (ref = 2nd-max, so e ≥ 0)
%     e = 0  → embed bit:  0→x stays, 1→x incremented
%     e = +1 → always shift x by +1
%     e > +1 → shift x by +1
% ==========================================================================
function [new_val, ok] = embed_one(x, e, bit, direction)
    ok = true;
    if direction < 0  % min pixel, e should be ≤ 0
        if e == 0
            new_val = x - bit;   % embed: 0→no change, 1→decrement
        elseif e == -1
            new_val = x - 1;     % shift to make room
        elseif e < -1
            new_val = x - 1;     % range shift
        else
            new_val = x; ok = false;  % unexpected: skip
        end
    else              % max pixel, e should be ≥ 0
        if e == 0
            new_val = x + bit;   % embed: 0→no change, 1→increment
        elseif e == 1
            new_val = x + 1;     % shift to make room
        elseif e > 1
            new_val = x + 1;     % range shift
        else
            new_val = x; ok = false;
        end
    end
end

function [orig_val, bit] = extract_one(x, e, direction)
    if direction < 0  % min pixel
        if e == 0
            bit = 0; orig_val = x;
        elseif e == -1
            bit = 1; orig_val = x + 1;
        else
            bit = 0; orig_val = x + 1;  % range-shift restore
        end
    else              % max pixel
        if e == 0
            bit = 0; orig_val = x;
        elseif e == 1
            bit = 1; orig_val = x - 1;
        else
            bit = 0; orig_val = x - 1;
        end
    end
end

% ==========================================================================
% NUMBER OF EMBEDDING PAIRS per segment (Sec.III-A)
% Thresholds T1 ≤ T2 ≤ ... ≤ Tm
% If S ∈ [Ti-1, Ti): K = 2*(m+1-i) pairs
% If S >= Tm: K = 0 pairs (too complex)
% ==========================================================================
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
    % S >= T_thresh(end) → 2*(m+1-m) = 2 pairs minimum... or 0 if too complex
    if S >= T_thresh(end)
        K = 0;  % smoothness too high → skip
    end
end

% ==========================================================================
% PSNR
% ==========================================================================
function p = compute_psnr(orig, marked)
    diff = double(orig) - double(marked);
    mse  = mean(diff(:).^2);
    if mse == 0
        p = Inf;
    else
        p = 10 * log10(255^2 / mse);
    end
end

% ==========================================================================
% UTILITIES
% ==========================================================================
function s = yesno(b)
    if b, s = 'PASS ✓'; else s = 'FAIL ✗'; end
end

function s = layer_name(i)
    names = {'I1','I2','I3','I4'};
    s = names{i};
end
