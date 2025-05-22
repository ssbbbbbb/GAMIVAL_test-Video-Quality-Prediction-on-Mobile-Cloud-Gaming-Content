data_name = 'LIVE-Meta-Mobile-Cloud-Gaming';
base_path = 'E:/FOLDER/computer_vision/GAMIVAL-main/feat_files/';

cnn_file = [base_path, data_name, '_CNN_bicubic_feats.mat'];
nss_file = [base_path, data_name, '_NSS_bicubic_feats.mat'];

if exist(cnn_file, 'file')
    load(cnn_file)
    f = feats_mat;
else
    error(['CNN file not found: ', cnn_file])
end

if exist(nss_file, 'file')
    load(nss_file)
    feats_mat = [feats_mat f];
else
    error(['NSS file not found: ', nss_file])
end

save([base_path, data_name, '_GAMIVAL_feats.mat'], 'feats_mat');
