import os

# soldier
# longdress
# loot
# redandblack

path = "/home/jupyter-austin2/zx/PointCloudUpsampling_original/tmp/rec/"
files = os.listdir(path)
files.sort()
frame=1000

for cls_name in files:
    
    old = os.path.join(path, cls_name)

    newname = '1_' + cls_name
    new = os.path.join(path, newname)
    print(new)
#     postfix = cls_name.split('_')[-1]
#     print(postfix)
# #     file_name = cls_name.split('.')[0].split('_')[:-1]
# #     block_num = cls_name.split('.')[0].split('_')[-1]
# #     file_name.append('qp42')
# #     file_name.append(block_num)
# #     temp = 'owlii3_vox11_' + str(frame)
#     print(cls_name.split('_')[:-1])
# #     temp = '_'.join(cls_name.split('_')[:-2])
#     temp = 'WPC3'
#     print(temp)
#     newname = temp + '_vox10_' + postfix
#     new = os.path.join(path, newname)
    os.rename(old, new)
#     frame += 1
#     frame += 1

