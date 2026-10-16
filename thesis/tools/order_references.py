"""Order numerical bibliography entries by their first citation in the manuscript."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
TOKEN=re.compile(r'\\(?:input|include)\{([^}]+)\}|\\cite\w*(?:\[[^\]]*\]){0,2}\{([^}]+)\}')

def citation_order():
    ordered=[]
    def visit(path,stack=()):
        assert path not in stack,f'Recursive TeX input: {path}'
        source=re.sub(r'(?<!\\)%[^\n]*','',path.read_text())
        for token in TOKEN.finditer(source):
            if token.group(1):
                child=path.parent/token.group(1)
                if child.suffix!='.tex':child=child.with_suffix('.tex')
                if child.name!='references.tex':visit(child,(*stack,path))
            else:
                for key in token.group(2).split(','):
                    key=key.strip()
                    if key not in ordered:ordered.append(key)
    visit(ROOT/'thesis.tex')
    return ordered

def main():
    path=ROOT/'references.tex';source=path.read_text()
    matches=list(re.finditer(r'(?m)^\\bibitem\{([^}]+)\}',source))
    end=source.index(r'\end{thebibliography}')
    blocks={m.group(1):source[m.start():matches[i+1].start() if i+1<len(matches) else end]
            for i,m in enumerate(matches)}
    order=citation_order()
    assert len(blocks)==len(matches) and set(order)==set(blocks)
    path.write_text(source[:matches[0].start()]+''.join(blocks[key] for key in order)+source[end:])
    report={'numbering':'first citation in document order','reference_count':len(order),
            'ordered_keys':order,'numbers':{key:i+1 for i,key in enumerate(order)}}
    (ROOT/'generated/citation_order.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Ordered {len(order)} references by first citation; first entry: {order[0]}.')

if __name__=='__main__':main()
