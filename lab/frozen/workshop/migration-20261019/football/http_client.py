"""CPU transport adapter. Image input is already encoded from actual rendering.

The Newton process may encode with its existing image library; it does not need
this package's OpenCV dependency. No simulator state is read by this adapter.
"""
import base64,json,urllib.request

def step(png_bytes,proprio,tof_frame,*,now_s,detections=None,base_url='http://127.0.0.1:8038',timeout_s=3.):
    payload={'rgb_png_base64':base64.b64encode(png_bytes).decode('ascii'),
             'proprio':proprio,'tof_frame':tof_frame,'now_s':now_s}
    if detections is not None:payload['detections']=detections
    request=urllib.request.Request(base_url+'/step',data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
    # Caller must catch network/schema failure and request zero velocity. Never
    # keep replaying the last nonzero command after the sensor service fails.
    with urllib.request.urlopen(request,timeout=timeout_s) as response:
        result=json.load(response)
    if result.get('schema')!='microduck.beta-control.v1':raise ValueError('Unexpected controller response')
    if result.get('runtime_id')!='newton-bam-20261019' or result.get('hardware_profile_id')!='beta-gc2093-l8cx-v1':raise ValueError('Wrong runtime/profile')
    return result
