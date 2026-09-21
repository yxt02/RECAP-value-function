#!/usr/bin/env python3
"""Read-only source conversion to browser-compatible VP9 preview, same frames/fps."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from fractions import Fraction
import json
from pathlib import Path
import time
import av


def convert(root,ep):
    source=root/'videos'/f"{ep['slug']}.mp4";target=root/'previews'/f"{ep['slug']}.webm"
    record=target.with_suffix('.json')
    identity=dict(source=str(source.resolve()),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
                  frames=ep['frames'],fps=ep['fps'],codec='libvpx-vp9',crf=32,cpu_used=8)
    if target.exists() and record.exists() and json.loads(record.read_text())['identity']==identity:
        return dict(slug=ep['slug'],reused=True)
    temporary=target.with_suffix('.webm.tmp');start=time.perf_counter()
    with av.open(str(source)) as inp,av.open(str(temporary),'w',format='webm') as out:
        stream=inp.streams.video[0];stream.codec_context.thread_count=1
        encoder=out.add_stream('libvpx-vp9',rate=Fraction(str(ep['fps'])))
        encoder.width=stream.width;encoder.height=stream.height;encoder.pix_fmt='yuv420p'
        encoder.codec_context.thread_count=2;encoder.bit_rate=0
        encoder.options={'deadline':'realtime','cpu-used':'8','crf':'32','row-mt':'1'}
        count=0
        for frame in inp.decode(stream):
            frame=frame.reformat(format='yuv420p');frame.pts=count;frame.time_base=1/Fraction(str(ep['fps']))
            for packet in encoder.encode(frame):out.mux(packet)
            count+=1
        for packet in encoder.encode():out.mux(packet)
    if count!=ep['frames']:
        raise ValueError(f"{ep['slug']}: source video frames={count}, expected={ep['frames']}")
    temporary.replace(target)
    result=dict(identity=identity,seconds=time.perf_counter()-start,bytes=target.stat().st_size)
    record.write_text(json.dumps(result,indent=2)+'\n')
    return dict(slug=ep['slug'],frames=count,seconds=result['seconds'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--workers',type=int,default=4);args=p.parse_args(argv)
    root=(Path(__file__).resolve().parents[2]/args.directory).resolve();(root/'previews').mkdir(exist_ok=True)
    eps=json.loads((root/'episodes.json').read_text());results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(convert,root,e) for e in eps]):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
    (root/'video_previews.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
