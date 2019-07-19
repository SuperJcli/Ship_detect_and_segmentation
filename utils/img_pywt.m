close all
clear all
clc

path = 'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment\data\';
save_path = 'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment\data_pywt\';
files = dir(path);
for i = 3:length(files)
    img = double(imread(strcat(path,files(i).name)));
    name_split = strsplit(files(i).name,'.');
    save_name = name_split{1};
    
    [C,S] = wavedec2(img,2,'haar');
    
    A2 = wrcoef2('a',C,S,'haar',2);
    A1 = wrcoef2('a',C,S,'haar',1);
    
    H1 = wrcoef2('h',C,S,'haar',1);
    V1 = wrcoef2('v',C,S,'haar',1);
    
    D1 = wrcoef2('d',C,S,'haar',1);
    H2 = wrcoef2('h',C,S,'haar',2);
    
    V2 = wrcoef2('v',C,S,'haar',2);
    D2 = wrcoef2('d',C,S,'haar',2);
    
    imwrite(uint16(img),strcat(save_path,save_name,'o.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(A1),strcat(save_path,save_name,'a.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(H1),strcat(save_path,save_name,'b.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(V1),strcat(save_path,save_name,'c.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(D1),strcat(save_path,save_name,'d.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(A2),strcat(save_path,save_name,'e.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(H2),strcat(save_path,save_name,'f.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(V2),strcat(save_path,save_name,'g.tif'),'Resolution',96,'Compression','lzw');
    imwrite(uint16(D2),strcat(save_path,save_name,'h.tif'),'Resolution',96,'Compression','lzw');
        
end