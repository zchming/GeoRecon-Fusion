"""External reference formats only. No fitting or choice based on reconstruction scores."""
import numpy as np
from ..geometry.camera import Camera


def rounded_rotation(matrix):
    """Canonicalize decimal-rounded calibration rotations, never estimated poses."""
    r=np.asarray(matrix,float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or np.max(np.abs(r.T@r-np.eye(3)))>1e-4 or abs(np.linalg.det(r)-1)>1e-4:
        raise ValueError('reference rotation is not a rounded proper rotation')
    u,_,vt=np.linalg.svd(r);out=u@vt
    if np.linalg.det(out)<0:raise ValueError('reference contains reflection')
    return out,float(np.max(np.abs(out-r)))


def strecha_camera(text):
    """OpenMVG io_readGTStrecha.hpp: P=K[R_file.T|-R_file.T*C]."""
    values=np.fromstring(text,sep=' ')
    if len(values)!=26 or not np.isfinite(values).all():raise ValueError('Strecha camera requires26 finite numbers')
    if np.any(values[9:12]!=0):raise ValueError('unsupported nonzero distortion metadata')
    width,height=values[24:26]
    if width!=int(width) or height!=int(height):raise ValueError('integer dimensions required')
    r,correction=rounded_rotation(values[12:21].reshape(3,3).T)
    center=values[21:24];camera=Camera(values[:9].reshape(3,3),r,-r@center,int(width),int(height))
    return camera,correction


def middlebury_cameras(text,width=640,height=480):
    """Official name_par.txt: filename K9 R9 t3, P=K[R|t]."""
    lines=[line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not lines or len(lines[0])!=1:raise ValueError('missing camera count')
    expected=int(lines[0][0]);result={}
    for row in lines[1:]:
        if len(row)!=22 or row[0] in result:raise ValueError('invalid or duplicate camera row')
        values=np.array([float(v) for v in row[1:]])
        if not np.isfinite(values).all():raise ValueError('nonfinite reference')
        r,correction=rounded_rotation(values[9:18].reshape(3,3))
        result[row[0]]=(Camera(values[:9].reshape(3,3),r,values[18:21],width,height),correction)
    if len(result)!=expected:raise ValueError('camera count mismatch')
    return result


def preprocess(rgb,camera,blur=False):
    from PIL import Image,ImageFilter
    scale=min(1.,800/camera.width,600/camera.height)
    width,height=max(1,round(camera.width*scale)),max(1,round(camera.height*scale))
    if rgb.size!=(camera.width,camera.height):raise ValueError('image/calibration dimensions differ')
    if blur:rgb=rgb.filter(ImageFilter.GaussianBlur(4))
    rgb=rgb.resize((width,height),Image.Resampling.BILINEAR)
    sx,sy=width/camera.width,height/camera.height
    transform=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    return rgb,transform@camera.K,transform
