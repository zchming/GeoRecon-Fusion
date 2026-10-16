"""Bounded retrieval of official DTU reference geometry, separate from inference."""
import argparse, zipfile
from pathlib import Path
from georecon.data.remote_zip import RangeZip
from georecon.data.protocol import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]
URL='https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip'

def fetch(scan):
    dest=ROOT/'data/stage5_point_gt'/f'scan{scan}'
    if dest.exists(): raise ValueError('existing download directory; preserve it')
    dest.mkdir(parents=True)
    r=RangeZip(URL,max_bytes=96*1024**2)
    assert r.size==6905656531 and r.etag=='"19b9bf4d3-5652d3a88539d"'
    names=[f'SampleSet/MVS Data/Points/stl/stl{scan:03}_total.ply',f'SampleSet/MVS Data/ObsMask/ObsMask{scan}_10.mat',f'SampleSet/MVS Data/ObsMask/Plane{scan}.mat']
    files=[]
    try:
        with zipfile.ZipFile(r) as z:
            for name in names:
                p=dest/name.rsplit('/',1)[1]
                p.write_bytes(z.read(name))
                files.append({'member':name,'local':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size})
        write_json(dest/'provenance.json',{'url':URL,'size':r.size,'etag':r.etag,'transferred_bytes':r.transferred,'ranges':r.ranges,'files':files})
    except Exception as e:
        write_json(dest/'failure.json',{'type':type(e).__name__,'message':str(e),'transferred_bytes':r.transferred})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scan',type=int,choices=[1,6],required=True);fetch(p.parse_args().scan)
