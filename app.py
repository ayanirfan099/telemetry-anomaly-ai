import argparse,json
from model import TelemetryModel,ROOT
from api import serve

def main():
    parser=argparse.ArgumentParser(description='Score fictional service telemetry for anomalies')
    parser.add_argument('command',choices=['evaluate','predict','serve'])
    parser.add_argument('--port',type=int,default=8003)
    parser.add_argument('--json',help='JSON array of telemetry records')
    args=parser.parse_args();model=TelemetryModel()
    if args.command=='serve':serve(lambda body:model.predict_batch(body.get('records')),args.port)
    elif args.command=='predict':
        if not args.json:parser.error('--json is required for predict')
        print(json.dumps(model.predict_batch(json.loads(args.json)),indent=2))
    else:
        report=model.evaluate();out=ROOT/'reports/evaluation.json';out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
