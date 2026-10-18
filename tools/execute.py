from pathlib import Path
import os, sys, json, shutil, base64, hashlib, time, subprocess
ROOT=Path(__file__).resolve().parents[2]; SITE=ROOT/'dashboard'; BUILD=SITE/'.build'
BUILD.mkdir(exist_ok=True)
for key, folder in [('JUPYTER_CONFIG_DIR','jupyter-config'),('JUPYTER_DATA_DIR','jupyter-data'),('JUPYTER_RUNTIME_DIR','jupyter-runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','mpl'),('TEMP','temp'),('TMP','temp')]:
    p=BUILD/folder;p.mkdir(exist_ok=True);os.environ[key]=str(p)
os.environ['R_LIBS_USER']=str(ROOT/'.R-library')
os.environ['PYTHONIOENCODING']='utf-8'
kp=BUILD/'jupyter-data/kernels/python3';kp.mkdir(parents=True,exist_ok=True)
(kp/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Concrete Python','language':'python'}))
os.environ['JUPYTER_PATH']=str(BUILD/'jupyter-data')

def run(number,lang):
    start=time.time(); key=f'tema{number}_{lang}'; folder=BUILD/key;folder.mkdir(exist_ok=True)
    shutil.copy2(SITE/'downloads/Concrete_Data.csv',folder/'Concrete_Data.csv')
    ext='ipynb' if lang=='Python' else 'R'
    src=SITE/'downloads'/f'{key}.{ext}';shutil.copy2(src,folder/src.name)
    print('INICIO',key,flush=True)
    if lang=='Python':
        import nbformat
        from nbclient import NotebookClient
        n=nbformat.read(src,as_version=4)
        NotebookClient(n,timeout=2400,kernel_name='python3',resources={'metadata':{'path':str(folder)}}).execute()
        nbformat.write(n,src)
        blocks=[]; title='Preparación'; prose=''
        for c in n.cells:
            if c.cell_type=='markdown':
                prose=c.source;lines=[x.strip('# *') for x in prose.splitlines() if x.strip()]
                if lines:title=lines[0]
                continue
            outputs=[]
            for o in c.outputs:
                if o.output_type=='error':raise RuntimeError(str(o))
                if o.output_type=='stream':outputs.append({'type':'text','value':o.text})
                else:
                    d=o.get('data',{})
                    if 'image/png' in d:
                        name=f'{key}-{len(blocks)}-{len(outputs)}.png';(SITE/'assets'/name).write_bytes(base64.b64decode(d['image/png']))
                        outputs.append({'type':'image','value':'assets/'+name})
                    elif 'text/html' in d:outputs.append({'type':'html','value':d['text/html']})
                    elif 'text/plain' in d:outputs.append({'type':'text','value':d['text/plain']})
            blocks.append({'title':title,'prose':prose,'code':c.source,'outputs':outputs})
    else:
        renv=os.environ.copy()
        for k in ['TEMP','TMP','TMPDIR']:renv[k]=os.environ.get('R_TEMP_DIR_DASHBOARD',str(BUILD/'temp'))
        result=subprocess.run(['C:/Program Files/R/R-4.6.1/bin/Rscript.exe',str(SITE/'tools/execute_R.R'),str(folder/src.name),str(folder)],env=renv,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (folder/'execution.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError(result.stdout+'\n'+result.stderr)
        blocks=json.loads((folder/'blocks.json').read_text(encoding='utf-8'))
        for bi,b in enumerate(blocks):
            for oi,o in enumerate(b['outputs']):
                if o['type']=='image':
                    path=Path(o['value']);name=f'{key}-{bi}-{oi}.svg';shutil.copy2(path,SITE/'assets'/name);o['value']='assets/'+name
    for f in folder.iterdir():
        if f.suffix in ['.csv','.txt','.svg'] and f.name!='Concrete_Data.csv':
            name=f'{key}_{f.name}';shutil.copy2(f,SITE/'downloads'/name)
            if f.name=='arbol_completo_python.svg':blocks[-1]['outputs'].append({'type':'image','value':'downloads/'+name,'wide':True})
    result={'topic':number,'language':lang,'blocks':blocks,'executed':'2026-10-18','duration':round(time.time()-start,2),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    (SITE/'data'/f'{key}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    print('OK',key,len(blocks),'bloques',result['duration'],'s',flush=True)

if __name__=='__main__':
    lang=sys.argv[1]
    for number in ([int(sys.argv[2])] if len(sys.argv)>2 else range(1,6)):run(number,lang)
