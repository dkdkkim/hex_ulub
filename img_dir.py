import os
import json
import numpy as np

# PATH = './'
json_dir = '/data/dk/ulub/dataset/utkface/'

var_list = [0.1, 0.2]

# folder_list = os.listdir(PATH)

total_img_list_train = []
total_img_list_val = []
total_img_list_val_bias = []

# gender_age

# with open(json_dir+'new_young_female.json', 'r') as file:
#    yf = json.load(file)
# with open(json_dir+'new_old_female.json', 'r') as file:
#    of = json.load(file)
# with open(json_dir+'new_old_male.json', 'r') as file:
#    om = json.load(file)
# with open(json_dir+'new_young_male.json', 'r') as file:
#    ym = json.load(file)
# with open(json_dir + 'white_female.json', 'r') as file:
#    wf = json.load(file)
# with open(json_dir + 'white_male.json', 'r') as file:
#    wm = json.load(file)
# with open(json_dir + 'black_female.json', 'r') as file:
#    bf = json.load(file)
# with open(json_dir + 'black_male.json', 'r') as file:
#    bm = json.load(file)
# with open(json_dir + 'asian_female.json', 'r') as file:
#    af = json.load(file)
# with open(json_dir + 'asian_male.json', 'r') as file:
#    am = json.load(file)
# with open(json_dir + 'indian_female.json', 'r') as file:
#    indf = json.load(file)
# with open(json_dir + 'indian_male.json', 'r') as file:
#    indm = json.load(file)
#
# with open(json_dir+'new_gender_age_test.json', 'r') as file:
#    test_list = json.load(file)
#
# yf_len, of_len, ym_len, om_len = len(yf), len(of), len(ym), len(om)
#
# yf_back, yf_data = yf[:int(0.2 * of_len)], yf[int(0.2 * of_len):]
# of_back, of_data = of[:int(0.2 * yf_len)], of[int(0.2 * yf_len):]
# ym_back, ym_data = ym[:int(0.2 * om_len)], ym[int(0.2 * om_len):]
# om_back, om_data = om[:int(0.2 * ym_len)], om[int(0.2 * ym_len):]
#
# file_list = yf_data + of_back[:int(len(yf_data)*var)] + \
#            om_data + ym_back[:int(len(om_data)*var)]
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# eb1_img_list  = [path + ' 1' if  path.split('/')[-1] in yf or path.split('/')[-1] in of else path + ' 0' for path in _img_list]
# print(f"[DEBUG] eb1_img_list: {len(eb1_img_list)}")
#
# file_list = of_data + yf_back[:int(len(of_data)*var)] + \
#            ym_data + om_back[:int(len(ym_data)*var)]
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# eb2_img_list  = [path + ' 1' if  path.split('/')[-1] in yf or path.split('/')[-1] in of else path + ' 0' for path in _img_list]
# print(f"[DEBUG] eb2_img_list: {len(eb2_img_list)}")
#
# file_list = test_list
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# test_img_list  = [path + ' 0' if  path.split('_')[1] == '0' else path + ' 1' for path in _img_list]
# print(f"[DEBUG] test_img_list: {len(test_img_list)}")
#
#
# with open('./eb1_' + str(var).split('.')[-1] + '.txt', 'w') as file_handler:
#    for item in eb1_img_list:
#        file_handler.write(f"{item}\n")
#
# with open('./eb2_' + str(var).split('.')[-1] + '.txt', 'w') as file_handler:
#    for item in eb2_img_list:
#        file_handler.write(f"{item}\n")
#
# with open('./test.txt', 'w') as file_handler:
#    for item in test_img_list:
#        file_handler.write(f"{item}\n")


# gender_race or race_gender
for var in var_list:
    with open(json_dir + 'race_gender_test_V2.json', 'r') as file:
        test_list = json.load(file)
    with open(json_dir + 'white_female_V2.json', 'r') as file:
        wf = json.load(file)
    with open(json_dir + 'white_male_V2.json', 'r') as file:
        wm = json.load(file)
    with open(json_dir + 'black_female_V2.json', 'r') as file:
        bf = json.load(file)
    with open(json_dir + 'black_male_V2.json', 'r') as file:
        bm = json.load(file)
    with open(json_dir + 'asian_female_V2.json', 'r') as file:
        af = json.load(file)
    with open(json_dir + 'asian_male_V2.json', 'r') as file:
        am = json.load(file)
    with open(json_dir + 'indian_female_V2.json', 'r') as file:
        indf = json.load(file)
    with open(json_dir + 'indian_male_V2.json', 'r') as file:
        indm = json.load(file)

    wf_len, wm_len, bf_len, bm_len, af_len, am_len, indf_len, indm_len \
        = len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(indf), len(indm)

    wf_back, wf_data = wf[:int(var * wm_len)], wf[int(var * wm_len):]
    wm_back, wm_data = wm[:int(var * wf_len)], wm[int(var * wf_len):]
    bf_back, bf_data = bf[:int(var * bm_len)], bf[int(var * bm_len):]
    bm_back, bm_data = bm[:int(var * bf_len)], bm[int(var * bf_len):]
    af_back, af_data = af[:int(var * am_len)], af[int(var * am_len):]
    am_back, am_data = am[:int(var * af_len)], am[int(var * af_len):]
    indf_back, indf_data = indf[:int(var * indm_len)], indf[int(var * indm_len):]
    indm_back, indm_data = indm[:int(var * indf_len)], indm[int(var * wf_len):]

    file_list = wf_data + wm_back[:int(len(wf_data) * var)] + \
                bm_data + bf_back[:int(len(bm_data) * var)] + \
                am_data + af_back[:int(len(am_data) * var)] + \
                indm_data + indf_back[:int(len(indm_data) * var)]
    _img_list = [os.path.join('utkcropped', file) for file in file_list]
    eb1_img_list = [path + ' 0' if path.split('/')[-1] in wf or path.split('/')[-1] in bf or path.split('/')[-1] in af or
                                   path.split('/')[-1] in indf else path + ' 1' for path in _img_list]
    # eb1_img_list  = [path + ' 0' if  path.split('/')[-1] in wf or path.split('/')[-1] in wm else path + ' 1' for path in _img_list]
    print(f"[DEBUG] eb1_img_list: {len(eb1_img_list)}")

    file_list = wm_data + wf_back[:int(len(wm_data) * var)] + \
                bf_data + bm_back[:int(len(bf_data) * var)] + \
                af_data + am_back[:int(len(af_data) * var)] + \
                indf_data + indm_back[:int(len(indf_data) * var)]
    _img_list = [os.path.join('utkcropped', file) for file in file_list]
    eb2_img_list = [path + ' 0' if path.split('/')[-1] in wf or path.split('/')[-1] in bf or path.split('/')[-1] in af or
                                   path.split('/')[-1] in indf else path + ' 1' for path in _img_list]
    # eb2_img_list  = [path + ' 0' if  path.split('/')[-1] in wf or path.split('/')[-1] in wm else path + ' 1' for path in _img_list]
    print(f"[DEBUG] eb2_img_list: {len(eb2_img_list)}")

    file_list = test_list
    _img_list = [os.path.join('utkcropped', file) for file in file_list]
    test_img_list = [path + ' 0' if path.split('_')[1] == '0' else path + ' 1' for path in _img_list]
    # test_img_list  = [path + ' 0' if  path.split('_')[2] == '0' else path + ' 1' for path in _img_list]
    print(f"[DEBUG] test_img_list: {len(test_img_list)}")

    with open(json_dir+f"eb1_gr_{str(var).split('.')[-1]}.txt", 'w') as file_handler:
        for item in eb1_img_list:
            file_handler.write(f"{item}\n")

    with open(json_dir+f'eb2_gr_' + str(var).split('.')[-1] + '.txt', 'w') as file_handler:
        for item in eb2_img_list:
            file_handler.write(f"{item}\n")

    with open(json_dir+'test_gr.txt', 'w') as file_handler:
        for item in test_img_list:
            file_handler.write(f"{item}\n")

# race_age
#
# with open(json_dir + 'race_age_test.json', 'r') as file:
#    test_list = json.load(file)
# with open(json_dir + 'old_white.json', 'r') as file:
#    ow = json.load(file)
# with open(json_dir + 'young_white.json', 'r') as file:
#    yw = json.load(file)
# with open(json_dir + 'old_black.json', 'r') as file:
#    ob = json.load(file)
# with open(json_dir + 'young_black.json', 'r') as file:
#    yb = json.load(file)
# with open(json_dir + 'old_asian.json', 'r') as file:
#    oa = json.load(file)
# with open(json_dir + 'young_asian.json', 'r') as file:
#    ya = json.load(file)
# with open(json_dir + 'old_indian.json', 'r') as file:
#    oi = json.load(file)
# with open(json_dir + 'young_indian.json', 'r') as file:
#    yi = json.load(file)
#
# ow_len, yw_len, ob_len, yb_len, oa_len, ya_len, oi_len, yi_len \
#    = len(ow), len(yw), len(ob), len(yb), len(oa), len(ya), len(oi), len(yi)
#
# ow_back, ow_data = ow[:int(yw_len*0.2)], ow[int(yw_len*0.2):]
# yw_back, yw_data = yw[:int(ow_len*0.2)], yw[int(ow_len*0.2):]
# ob_back, ob_data = ob[:int(yb_len*0.2)], ob[int(yb_len*0.2):]
# yb_back, yb_data = yb[:int(ob_len*0.2)], yb[int(ob_len*0.2):]
# oa_back, oa_data = oa[:int(ya_len*0.2)], oa[int(ya_len*0.2):]
# ya_back, ya_data = ya[:int(oa_len*0.2)], ya[int(oa_len*0.2):]
# oi_back, oi_data = oi[:int(yi_len*0.2)], oi[int(yi_len*0.2):]
# yi_back, yi_data = yi[:int(oi_len*0.2)], yi[int(oi_len*0.2):]
#
#
# file_list = yw_data + ow_back[:int(len(yw_data)*var)] + \
#            ob_data + yb_back[:int(len(ob_data)*var)] + \
#            oa_data + ya_back[:int(len(oa_data)*var)] + \
#            oi_data + yi_back[:int(len(oi_data)*var)]
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# eb1_img_list  = [path + ' 0' if  path.split('/')[-1] in ow or path.split('/')[-1] in yw else path + ' 1' for path in _img_list]
# print(f"[DEBUG] eb1_img_list: {len(eb1_img_list)}")
#
#
# file_list = ow_data + yw_back[:int(len(ow_data)*var)] + \
#            yb_data + ob_back[:int(len(yb_data)*var)] + \
#            ya_data + oa_back[:int(len(ya_data)*var)] + \
#            yi_data + oi_back[:int(len(yi_data)*var)]
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# eb2_img_list  = [path + ' 0' if  path.split('/')[-1] in ow or path.split('/')[-1] in yw else path + ' 1' for path in _img_list]
# print(f"[DEBUG] eb2_img_list: {len(eb2_img_list)}")
#
# file_list = test_list
# _img_list = [os.path.join('utkcropped', file) for file in file_list]
# test_img_list  = [path + ' 0' if  path.split('_')[2] == '0' else path + ' 1' for path in _img_list]
# print(f"[DEBUG] test_img_list: {len(test_img_list)}")
#
# with open('./eb1_ra_' + str(var).split('.')[-1] + '.txt', 'w') as file_handler:
#    for item in eb1_img_list:
#        file_handler.write(f"{item}\n")
#
# with open('./eb2_ra_' + str(var).split('.')[-1] + '.txt', 'w') as file_handler:
#    for item in eb2_img_list:
#        file_handler.write(f"{item}\n")
#
# with open('./test_ra.txt', 'w') as file_handler:
#    for item in test_img_list:
#        file_handler.write(f"{item}\n")
