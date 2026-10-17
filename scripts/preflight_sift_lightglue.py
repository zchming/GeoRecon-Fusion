"""Standalone compatibility smoke; never invokes geometry, GT or fusion."""
import json, hashlib, time, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SIFT
from lightglue.sift import run_opencv_sift, sift_to_rootsift
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/sift_lightglue_preflight'
VENDOR=Path('/home/zchming/.cache/georecon/vendor/LightGlue-eb42fee')
WEIGHT=Path.home()/'.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(f):
    n=f['keypoints'].shape[1]
    assert f['keypoints'].shape==(1,n,2) and f['descriptors'].shape==(1,n,128)
    assert f['scales'].shape==f['oris'].shape==(1,n)
    assert f['image_size'].shape==(1,2) and (f['image_size']>0).all()
    assert all(torch.isfinite(v).all() for v in f.values())
    assert (f['scales']>0).all() and (f['oris']>=0).all() and (f['oris']<=2*np.pi).all()
    assert (f['keypoints']>=0).all() and (f['keypoints']<f['image_size'][:,None,:]).all()
    assert torch.allclose(f['descriptors'].norm(dim=-1),torch.ones(1,n),atol=1e-5)
    return n

def main():
    cv2.setNumThreads(8);cv2.setRNGSeed(42);cv2.ocl.setUseOpenCL(False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()=='eb42fee2d71449efb0aa5c10549752b5d75384d8'
    assert digest(WEIGHT)=='5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0'
    torch.set_num_threads(8);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(11*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    matcher=LightGlue(features='sift').eval().cuda()
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    for i in range(9):
        state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
    check=matcher.load_state_dict(state,strict=False)
    assert not set(check.missing_keys)&set(dict(matcher.named_parameters()))
    assert not check.unexpected_keys
    torch.cuda.synchronize();load_seconds=time.perf_counter()-start
    report={'scope':'API/resource preflight only; no backend, GT evaluation or formal comparison','weight_sha256':digest(WEIGHT),'vendor_commit':'eb42fee2d71449efb0aa5c10549752b5d75384d8','source_sha256':{str(p.relative_to(VENDOR)):digest(p) for p in [VENDOR/'lightglue/sift.py',VENDOR/'lightglue/lightglue.py',VENDOR/'lightglue/utils.py']},'versions':{'python':sys.version,'torch':torch.__version__,'opencv':cv2.__version__,'numpy':np.__version__},'gpu':torch.cuda.get_device_name(0),'allocator_budget_bytes':11*1024**3,'model_load_seconds':load_seconds,'model_load_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_load_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weight_missing_keys':check.missing_keys,'weight_unexpected_keys':check.unexpected_keys,'matcher_config':vars(matcher.conf),'cases':[]}
    for case in ['dtu_scan1_001_003','dtu_scan1_001_020']:
        base=ROOT/'data/stage6_pairs'/case;manifest=json.loads((base/'input.json').read_text());features=[];records=[]
        old=np.load(ROOT/'outputs/stage6_two_view_v1'/case/'features_matches.npz')
        for idx,item in enumerate(manifest['images']):
            path=base/item['path'];assert digest(path)==item['sha256'];start=time.perf_counter()
            gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            # Historical RGB->OpenCV gray, rather than alternate float RGB weights.
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            xy,scores,scales,angles,des=run_opencv_sift(cv2.SIFT_create(nfeatures=4096),gray)
            assert len(xy)<=4096 and np.array_equal(xy.astype(float),old[['keypoints_first','keypoints_second'][idx]])
            f={k:torch.from_numpy(v)[None] for k,v in {'keypoints':xy,'scales':scales,'oris':angles,'descriptors':des}.items()}
            f['descriptors']=sift_to_rootsift(f['descriptors']);f['image_size']=torch.tensor([[gray.shape[1],gray.shape[0]]],dtype=torch.float32)
            validate(f);records.append({'image_id':item['image_id'],'sha256':digest(path),'size_wh':[gray.shape[1],gray.shape[0]],'features':len(xy),'extract_and_adapt_seconds':time.perf_counter()-start,'historical_xy_exact_parity':True})
            features.append(f)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        gpu=[{k:v.cuda() for k,v in f.items()} for f in features];torch.cuda.synchronize();transfer=time.perf_counter()-start;start=time.perf_counter()
        with torch.inference_mode():out=matcher({'image0':gpu[0],'image1':gpu[1]})
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        pairs=out['matches'][0].cpu().numpy();scores=out['scores'][0].cpu().numpy();m0=out['matches0'][0].cpu().numpy();m1=out['matches1'][0].cpu().numpy()
        assert pairs.shape==(len(scores),2) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        for side in [0,1]:assert len(np.unique(pairs[:,side]))==len(pairs) and (pairs[:,side]>=0).all() and (pairs[:,side]<len(features[side]['keypoints'][0])).all()
        assert np.array_equal(m0[pairs[:,0]],pairs[:,1]) and np.array_equal(m1[pairs[:,1]],pairs[:,0])
        assert (m0>=0).sum()==len(pairs) and (m1>=0).sum()==len(pairs)
        local=ROOT/'outputs/sift_lightglue_preflight'/case;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'matches.npz',match_indices=pairs,scores=scores,pixels_first=features[0]['keypoints'][0].numpy()[pairs[:,0]],pixels_second=features[1]['keypoints'][0].numpy()[pairs[:,1]])
        report['cases'].append({'case_id':case,'inputs':records,'h2d_seconds':transfer,'matcher_seconds_single_pass':elapsed,'matches_contract_count_only':len(pairs),'stop_layer':int(out['stop']),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'whole_card_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip(),'index_and_mutuality_checks':True,'local_npz_sha256':digest(local/'matches.npz')})
        del gpu,out
    # Exercise official extractor separately, preserving original image size.
    image=cv2.cvtColor(cv2.imread(str(ROOT/'data/stage6_pairs/dtu_scan1_001_003/0.png')),cv2.COLOR_BGR2RGB)
    extractor=SIFT(backend='opencv').eval()
    with torch.inference_mode():f=extractor.extract(torch.from_numpy(image).permute(2,0,1).float()/255,resize=None)
    validate(f)
    assert torch.equal(f['image_size'],torch.tensor([[800.,600.]]))
    report['official_extractor_separate_contract']={'features':len(f['keypoints'][0]),'configuration':vars(extractor.conf),'resize_override':None,'image_size_wh':f['image_size'].tolist(),'not_used_for_matcher_smoke':True}
    blank=cv2.imread(str(ROOT/'data/stage6_pairs/blank_pair/0.png'),cv2.IMREAD_GRAYSCALE) if (ROOT/'data/stage6_pairs/blank_pair/0.png').exists() else np.zeros((600,800),np.uint8)
    kp,des=cv2.SIFT_create(nfeatures=4096).detectAndCompute(blank,None);assert not kp and des is None
    report['empty_descriptor_policy']='explicit skip before adapter/matcher; detector empty case checked; no geometry attempted'
    report['limitations']=['single pass timing includes cold kernels, not a benchmark','whole-card snapshot is not continuous peak measurement; allocator peak is measured','matches are unverified; no accuracy, pose or robustness gain established','official OpenCV coordinate TODO remains; no claim of pycolmap convention equivalence','legacy detector + RootSIFT adaptation changes descriptor metric; a future fair matcher comparison needs shared features']
    (DOC/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
