"""Run the unchanged estimated-K COLMAP adapter on Stage7B's exact processed images."""
import argparse
import json
from pathlib import Path
import shutil
from georecon.data.protocol import write_json,sha256
from georecon.baselines.colmap import run_colmap


def run(study,config,output):
    study=Path(study).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bundle=json.loads((study/'inference.json').read_text())
    for case in bundle['cases']:
        folder=output/case['id'];folder.mkdir();items=[]
        for image_id,key in zip(case['view_ids'],case['keys']):
            item=bundle['images'][key];path=folder/(image_id+'.png');shutil.copyfile(item['path'],path)
            items.append({'image_id':image_id,'path':path.name,'sha256':sha256(path),'width':800,'height':600})
        manifest={'schema_version':1,'kind':'real','scene_id':case['id'],'split':'dev' if case['split']=='development' else 'test','seed':42,
            'source':{'study_protocol_sha256':sha256(study/'protocol.json'),'exact_processed_inputs':True},
            'license_note':'DTU citeware, Jensen et al. CVPR2014; source data local only',
            'preprocessing':{'operation':'identity'},'budget':{'colmap_config':str(config)},'images':items}
        write_json(folder/'manifest.json',manifest)
        result=run_colmap(folder/'manifest.json',config,folder/'run')
        print(case['id'],result['status'],sum(i['registered'] for i in result['images']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['study','config','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.study,a.config,a.output)
