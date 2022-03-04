cls_type='gr'
#bias=1
#var=1
gpu='0'

#python run_utkface.py -c utkface_eb${bias}_var${var}_01 -eb $bias -var $var -cls $cls_type -gpu $gpu
#python run_utkface.py -c utkface_eb${bias}_var${var}_02 -eb $bias -var $var -cls $cls_type -gpu $gpu
#python run_utkface.py -c utkface_eb${bias}_var${var}_03 -eb $bias -var $var -cls $cls_type -gpu $gpu

bias=1
var=2
python run_utkface.py -c utkface_eb${bias}_var${var}_01 -eb $bias -var $var -cls $cls_type -gpu $gpu
python run_utkface.py -c utkface_eb${bias}_var${var}_02 -eb $bias -var $var -cls $cls_type -gpu $gpu
python run_utkface.py -c utkface_eb${bias}_var${var}_03 -eb $bias -var $var -cls $cls_type -gpu $gpu

#bias=2
#var=1
#python run_utkface.py -c utkface_eb${bias}_var${var}_01 -eb $bias -var $var -cls $cls_type -gpu $gpu
#python run_utkface.py -c utkface_eb${bias}_var${var}_02 -eb $bias -var $var -cls $cls_type -gpu $gpu
#python run_utkface.py -c utkface_eb${bias}_var${var}_03 -eb $bias -var $var -cls $cls_type -gpu $gpu

bias=2
var=2
python run_utkface.py -c utkface_eb${bias}_var${var}_01 -eb $bias -var $var -cls $cls_type -gpu $gpu
python run_utkface.py -c utkface_eb${bias}_var${var}_02 -eb $bias -var $var -cls $cls_type -gpu $gpu
python run_utkface.py -c utkface_eb${bias}_var${var}_03 -eb $bias -var $var -cls $cls_type -gpu $gpu

