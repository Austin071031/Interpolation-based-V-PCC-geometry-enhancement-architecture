'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''
def model_structure(model):
    blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4
    with open("model_structure.csv", 'a') as fl:
        for index, (key, w_variable) in enumerate(model.named_parameters()):
    #         if len(key) <= 20:
    #             key = key + (20 - len(key)) * blank
            shape = str(w_variable.shape)
    #         if len(shape) <= 30:
    #             shape = shape + (30 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
    #         if len(str_num) <= 10:
    #             str_num = str_num + (10 - len(str_num)) * blank
            line = ""
            print('| {} | {} | {} |'.format(key, shape, str_num))
            line = key + "," + str(str_num)
            fl.write(line + "\n")
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        fl.write("total parameters of model" + "," + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

