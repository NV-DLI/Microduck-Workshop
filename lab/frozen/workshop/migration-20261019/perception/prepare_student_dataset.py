#!/usr/bin/env python3
"""Copy the course-provided real Beta images into a fresh student workspace.

This is dataset preparation, NOT new student capture or detector training.
Every image/label is verified before copying; split membership stays fixed.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from train_detector import validate_dataset


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new student dataset directory')
    manifest=validate_dataset(source)
    if manifest.get('development_only'):raise ValueError('Course-provided data must contain all declared scenes')
    output.mkdir(parents=True)
    try:
        for row in manifest['images']:
            for kind in ('image','label'):
                target=output/row[kind];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/row[kind],target)
        for filename in ('collection-plan.json',):
            if (source/filename).is_file():shutil.copy2(source/filename,output/filename)
        (output/'dataset.yaml').write_text('path: '+json.dumps(str(output))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n  1: goal\n')
        provenance=dict(mode='course-provided-real-RTX-images',source_dataset=str(source),
            source_manifest_sha256=digest(source/'manifest.json'),student_capture_performed=False,
            student_training_performed=False,source_detector_weights_copied=False,
            description='Students train new random weights from these supplied images; this action only prepares verified files.')
        manifest.update(dataset_yaml_sha256=digest(output/'dataset.yaml'),student_dataset_provenance=provenance)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        checked=validate_dataset(output)
        result=dict(schema='microduck.student-detector-dataset-preparation.v1',status='completed',
            runtime_id=manifest['runtime_id'],hardware_profile_id=manifest['hardware_profile_id'],
            output=str(output),image_count=len(checked['images']),dataset_manifest_sha256=digest(output/'manifest.json'),
            **provenance)
        (output/'student-preparation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except BaseException as error:
        (output/'preparation-error.json').write_text(json.dumps({'status':'failed','error':repr(error)})+'\n')
        raise


if __name__=='__main__':main()
