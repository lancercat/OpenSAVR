import os.path

from neko_2022_soai_zero.visualization.result_compilers.bootstrap import accrfolder,detailed
from neko_sdk.ocr_modules.charset.jpn_filters import get_jpn_filters
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter,neko_oswr_Attention_AR_counter

def detail_folder(src):
    fd=get_jpn_filters();
    for fk in fd:
        rec = accrfolder(root=src, filter=fd[fk], dst=os.path.join(src,"details"),
                         thresh=[10, 8, 5, 3, 0],
                     arcntr=neko_os_Attention_AR_counter("Details-"+fk,False),
                         case_sensitive=False);
#detail_folder("/run/media/lasercat/ssddata/project283dump/open_basemodel_prec_shufIN3_g3kDPs_cycanof_aa/closeset_benchmarks/shuf3_chs_prototyper/JAP_lang/");
#detail_folder("/run/media/lasercat/ssddata/project283dump/open_basemodel/closeset_benchmarks/base_chs_prototyper/JAP_lang/");
detail_folder("/run/media/lasercat/ssddata/project283dump/open_basemodelXL512_prec_shufIN3_g3kDPs_cycanof_aa/closeset_benchmarks/shuf3_chs_prototyper/JAP_lang/");
