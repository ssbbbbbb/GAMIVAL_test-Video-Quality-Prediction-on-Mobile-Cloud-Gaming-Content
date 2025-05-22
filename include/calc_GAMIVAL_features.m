function feats_frames = calc_GAMIVAL_features(test_video, width, height, ...
                                              framerate, log_level)
    feats_frames = [];

    % ==== 無限等待直到 .yuv 檔案出現 ====
    pause(0.2);
    while true
        if exist(test_video, 'file')
            test_file = fopen(test_video, 'r');
            if test_file ~= -1
                break;
            end
        end
        pause(0.1);
    end

    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    nb_frames = floor(file_length / width / height / 1.5);

    % get features for each chunk
    for fr = floor(framerate/2):framerate:nb_frames-2
        % === 讀取前後幀 ===
        prev_YUV_frame = YUVread(test_file, [width height], max(1, fr - floor(framerate/3)));
        next_YUV_frame = YUVread(test_file, [width height], min(nb_frames - 2, fr + floor(framerate/3)));

        % === RGB 轉換並轉 GPU ===
        prev_rgb_gpu = gpuArray(ycbcr2rgb(uint8(prev_YUV_frame)));
        next_rgb_gpu = gpuArray(ycbcr2rgb(uint8(next_YUV_frame)));

        % === 轉換回 CPU（spatial 特徵不支援 GPU）===
        prev_rgb_cpu = gather(prev_rgb_gpu);
        next_rgb_cpu = gather(next_rgb_gpu);

        % === 尺寸轉換 ===
        sside = min(size(prev_YUV_frame,1), size(prev_YUV_frame,2));
        ratio = 540 / sside;
        if ratio < 1
            prev_rgb_cpu = imresize(prev_rgb_cpu, ratio);  % CPU 版本
            next_rgb_cpu = imresize(next_rgb_cpu, ratio);
            prev_rgb_gpu = imresize(prev_rgb_gpu, ratio);  % GPU 版本
            next_rgb_gpu = imresize(next_rgb_gpu, ratio);
        end

        feats_per_frame = [];

        %% === Spatial NSS 特徵（CPU） ===
        try
            if log_level == 1, fprintf('- Extracting Spatial NSS...'); tic; end
            prev_feats_spt = GAMIVAL_spatial_features(prev_rgb_cpu);
            next_feats_spt = GAMIVAL_spatial_features(next_rgb_cpu);
            feats_spt_mean = nanmean([prev_feats_spt; next_feats_spt]);
            if log_level == 1, toc; end
        catch ME
            warning('⚠️ Spatial feature extraction failed: %s', ME.message);
            feats_spt_mean = nan(1, 680);  % 預設維度
        end
        feats_per_frame = [feats_per_frame, feats_spt_mean];

        %% === Temporal NSS 特徵（GPU） ===
        if log_level == 1, fprintf('- Extracting temporal NSS...'); tic; end
        wfun = load(fullfile('include', 'WPT_Filters', 'haar_wpt_3.mat'));
        wfun = wfun.wfun;
        frames_wpt = gpuArray(zeros(size(prev_rgb_gpu,1), size(prev_rgb_gpu,2), size(wfun, 2)));

        fr_idx_start = max(1, fr - floor(size(wfun,2)/2));
        fr_idx_end = min(nb_frames - 3, fr_idx_start + size(wfun,2) - 1);
        fr_wpt_cnt = 1;

        for fr_wpt = fr_idx_start:fr_idx_end
            YUV_tmp = YUVread(test_file, [width height], fr_wpt);
            frameY = gpuArray(YUV_tmp(:,:,1));
            if ratio < 1
                frameY = imresize(frameY, ratio);  % GPU-safe imresize
            end
            frames_wpt(:,:,fr_wpt_cnt) = frameY;
            fr_wpt_cnt = fr_wpt_cnt + 1;
        end

        dpt_filt_frames = gpuArray(zeros(size(prev_rgb_gpu,1), size(prev_rgb_gpu,2), size(wfun,1)));
        feats_tmp_wpt = [];
        w = 1.5;

        for freq = 1:size(wfun, 1)
            filt = reshape(wfun(freq,:), 1, 1, []);
            dpt_filt_frames(:,:,freq) = sum(frames_wpt .* filt, 3) + ...
                gpuArray(randn(size(frames_wpt,1), size(frames_wpt,2)) * w);

            for scale = 1:2
                y_scale = imresize(dpt_filt_frames(:,:,freq), 2 ^ (-(scale - 1)));  % GPU
                tmp_feat = GAMIVAL_basic_extractor(y_scale);
                feats_tmp_wpt = [feats_tmp_wpt, gather(tmp_feat)];
            end
        end
        if log_level == 1, toc; end

        feats_per_frame = [feats_per_frame, feats_tmp_wpt];
        feats_frames(end+1,:) = feats_per_frame;
    end

    fclose(test_file);
end

%% === YUV frame 讀取 ===
function YUV = YUVread(f, dim, frnum)
    fseek(f, dim(1) * dim(2) * 1.5 * frnum, 'bof');

    Y = fread(f, dim(1) * dim(2), 'uchar');
    if length(Y) < dim(1) * dim(2), YUV = []; return; end
    Y = cast(reshape(Y, dim(1), dim(2)), 'double');

    U = fread(f, dim(1) * dim(2) / 4, 'uchar');
    if length(U) < dim(1) * dim(2) / 4, YUV = []; return; end
    U = cast(reshape(U, dim(1)/2, dim(2)/2), 'double');
    U = imresize(U, 2.0);  % CPU safe

    V = fread(f, dim(1) * dim(2) / 4, 'uchar');
    if length(V) < dim(1) * dim(2) / 4, YUV = []; return; end    
    V = cast(reshape(V, dim(1)/2, dim(2)/2), 'double');
    V = imresize(V, 2.0);  % CPU safe

    YUV(:,:,1) = Y';
    YUV(:,:,2) = U';
    YUV(:,:,3) = V';
end
