"""Stage final deliverables and a clean archive; never publishes externally."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parents[1]/'outputs/final_v6_2_package'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for stem,qa in [('manuscript','main_final_release'),('supplement','supp_final_release')]:
        name=f'MLST_HEA_{stem}_v6_2'
        shutil.copy2(ROOT/'qa'/qa/f'{name}.pdf',ROOT/'documents'/f'{name}.pdf')
        for suffix in ['.docx','.pdf']:shutil.copy2(ROOT/'documents'/f'{name}{suffix}',DEST/f'{name}{suffix}')
    shutil.copy2(ROOT/'V6_2_DUZELTME_NOTU_TR.md',DEST/'V6_2_DUZELTME_NOTU_TR.md')
    # Export exactly the active figures embedded in the final main Word file.
    d=Document(ROOT/'documents/MLST_HEA_manuscript_v6_2.docx')
    images=ROOT/'figures_main';images.mkdir(exist_ok=True)
    for i,shape in enumerate(d.inline_shapes,1):
        rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        (images/f'Figure_{i}.png').write_bytes(d.part.related_parts[rid].blob)
    excluded_scripts={'build_v6_documents.py','build_v6_1_documents.py'}
    paths=[]
    for directory in ['data','inputs','results','figures','figures_main','scripts','tests','independent_generator_R_v6_1','templates']:
        for file in sorted((ROOT/directory).rglob('*')):
            if not file.is_file() or '__pycache__' in file.parts or file.suffix=='.pyc':continue
            if directory=='scripts' and file.name in excluded_scripts:continue
            if directory=='templates' and file.name not in ['manuscript_v6_1.docx','supplement_v6_1.docx']:continue
            paths.append((file,file.relative_to(ROOT).as_posix()))
    for file in sorted((ROOT/'documents').glob('*v6_2.*')):paths.append((file,file.relative_to(ROOT).as_posix()))
    paths.append((ROOT/'README_v6_2.md','README.md'))
    paths.append((ROOT/'V6_2_DUZELTME_NOTU_TR.md','V6_2_DUZELTME_NOTU_TR.md'))
    manifest={relative:hashlib.sha256(file.read_bytes()).hexdigest() for file,relative in paths}
    manifest_path=ROOT/'SHA256_v6_2.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=DEST/'MLST_HEA_v6_2_complete_package.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file,relative in paths:z.write(file,'MLST_HEA_v6_2/'+relative)
        z.write(manifest_path,'MLST_HEA_v6_2/SHA256_v6_2.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path,sha in manifest.items():assert hashlib.sha256(z.read('MLST_HEA_v6_2/'+path)).hexdigest()==sha
    print(json.dumps({'archive':str(archive),'files':len(paths),'integrity_verified':True,'external_writes':False},indent=2))

if __name__=='__main__':main()
