"""Needle's public single-turn training prompt (no hidden product instructions)."""
import json

def normalize_tools(tools):
    if not isinstance(tools,list):
        raise ValueError('tools must be a list')
    out=[]
    for t in tools:
        if not isinstance(t,dict):raise ValueError('every tool must be a schema object')
        if t.get('type')=='function': t=t['function']
        if not isinstance(t.get('name'),str) or not t['name']:
            raise ValueError('every tool needs a name')
        out.append(t)
    return out

def render_prompt(query, tools, system=None):
    tools=normalize_tools(tools)
    prefix=f'<|im_start|>system\n{system}<|im_end|>\n' if system else ''
    return (prefix+'<|im_start|>user\n<tools>'+json.dumps(tools,ensure_ascii=False,separators=(',',':'))+
            '</tools>\n'+query+'<|im_end|>\n<|im_start|>assistant\n')

def parse_response(text):
    reasoning=''
    if '<think>' in text and '</think>' in text:
        reasoning=text.split('<think>',1)[1].split('</think>',1)[0].strip()
    calls=None
    error=None
    if '<tool_call>' in text and '</tool_call>' in text:
        payload=text.split('<tool_call>',1)[1].split('</tool_call>',1)[0]
        try:
            calls=json.loads(payload)
            if not isinstance(calls,list) or any(not isinstance(c,dict) or not isinstance(c.get('name'),str) or not isinstance(c.get('arguments'),dict) for c in calls):
                raise ValueError('invalid tool call shape')
        except (json.JSONDecodeError,ValueError) as exc:
            error=str(exc); calls=None
    else:
        error='generation did not contain a complete tool-call section'
    return dict(function_calls=calls,reasoning=reasoning,parse_error=error,text=text)
