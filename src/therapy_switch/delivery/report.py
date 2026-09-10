"""Self-contained browser report: no web server, CDN, plotting service or JS library."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


def write_client_report(targets, coverage, manifest, path):
    fields = [
        ("hcp_rank", "Rank"),
        ("hcp_id", "HCP ID"),
        ("specialty", "Specialty"),
        ("geography", "Geography"),
        ("organization", "Organization"),
        ("priority_patient_count", "Priority patients"),
        ("eligible_patient_count", "Eligible patients"),
        ("priority_share_percent", "Priority share (%)"),
        ("priority_tier", "Tier"),
    ]
    records = json.loads(targets.fillna("").to_json(orient="records"))
    payload = (
        json.dumps(records, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    )

    def escape(value):
        return html.escape(str(value), quote=True)

    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key, _ in fields) + "</tr>"
        for row in records
    )
    headings = "".join(
        f'<th><button data-sort="{key}">{title}</button></th>' for key, title in fields
    )
    metrics = manifest.get("historical_evaluation", {}).get("test", {})
    evaluation = "No labelled outcomes are used for the current scoring batch."
    if "recall" in metrics:
        evaluation += (
            f" A separate historical test set captured {metrics['recall']:.1%} of switchers "
            f"at {manifest['patient_fraction']:.0%} patient capacity "
            f"(precision {metrics['precision']:.1%}; AP {metrics['ap']:.3f}). "
            "This is a historical evaluation, not an outcome estimate for this list."
        )
    fraction = manifest["patient_fraction"]
    evidence = manifest["model_evidence"]
    evidence_text = "Experimental candidate — a significant improvement has not been established."
    if evidence == "reference":
        evidence_text = "Reference model — review performance before operational use."
    elif evidence != "experimental":
        evidence_text = f"Model evidence status supplied by the pipeline configuration: {evidence}."
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HCP opportunity review</title>
<style>
:root{--ink:#152d36;--muted:#526a73;--teal:#087c78;--light:#edf6f4;--line:#dce6e7}
*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:var(--ink);font:15px/1.5 Arial,sans-serif}
main{max-width:1420px;margin:auto;padding:38px 34px 50px}
header{display:flex;justify-content:space-between;gap:24px;align-items:start}
.kicker{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--teal);font-weight:bold}
h1{font-size:34px;margin:8px 0}h2{font-size:19px;margin:0 0 16px}p{margin:7px 0;color:var(--muted)}
.date{text-align:right;white-space:nowrap}.badge{display:inline-block;background:var(--light);border:1px solid #b3d9d2;border-radius:20px;padding:6px 13px}
.notice{margin:24px 0 18px;padding:13px 17px;background:#fff6e4;border-left:4px solid #c48a28;border-radius:5px;color:#654c1d}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:22px}
.card,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:23px}
.value{font-size:34px;line-height:1.15;font-weight:bold}.label{color:var(--muted);margin-top:9px}
.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:20px;margin-bottom:22px}
.bar{display:grid;grid-template-columns:115px 1fr 45px;align-items:center;gap:10px;margin:11px 0;font-size:13px}
.track{height:18px;background:#eef3f4;border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--teal)}
.method p{margin-bottom:15px}.small{font-size:12px;color:var(--muted)}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:18px 0}
label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
input,select,button{font:inherit}input,select{padding:9px 12px;border:1px solid #bfced0;border-radius:6px;background:white;min-width:170px}
.action{padding:10px 15px;border:0;border-radius:6px;background:var(--teal);color:white;cursor:pointer}
.table-wrap{overflow:auto;max-height:650px}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:#edf4f4;font-size:12px}th button{background:none;border:0;color:var(--ink);font-weight:bold;cursor:pointer;padding:0}
tbody tr:hover{background:#f3faf8}.count{margin-left:auto;color:var(--muted)}footer{margin-top:22px}
details{margin-top:14px}summary{cursor:pointer;font-weight:bold}
@media(max-width:850px){main{padding:22px 15px}.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}header{display:block}.date{text-align:left;margin-top:14px}h1{font-size:28px}}
@media print{main{padding:0}.controls{display:none}.table-wrap{max-height:none;overflow:visible}body{background:white}.panel,.card{break-inside:avoid}}
</style></head>
<body><main>
<header><div><div class="kicker">Field planning · advanced therapy</div><h1>HCP opportunity review</h1>
<p>Prioritize HCPs with eligible patients in the highest-scoring __FRACTION__ of the patient population.</p></div>
<div class="date"><div class="badge">__SCOPE__</div><p>Assessment: <strong>__DATE__</strong></p></div></header>
<div class="notice">__EVIDENCE__</div>
<section class="cards">
<div class="card"><div class="value">__ELIGIBLE__</div><div class="label">Eligible patients assessed</div></div>
<div class="card"><div class="value">__PRIORITY__</div><div class="label">Patients in the priority list</div></div>
<div class="card"><div class="value">__HCPS__</div><div class="label">HCPs with priority patients</div></div>
<div class="card"><div class="value">__ATTRIBUTED__</div><div class="label">Priority patients attributed</div></div>
</section>
<section class="grid"><div class="panel"><h2>Highest-priority HCPs</h2><div id="bars"></div>
<p class="small">Bars show priority patient counts for the first 12 HCPs in the current view.</p></div>
<div class="panel method"><h2>How to read this list</h2>
<p><strong>Priority patients</strong> are the top __FRACTION__ by patient risk score, selected once per patient on the assessment date.</p>
<p>Each patient is assigned to one relevant HCP using the configured attribution rule. HCPs are ranked by priority patient count, then priority share, with HCP ID breaking ties.</p>
<p><strong>__UNASSIGNED__ priority patients have no eligible HCP attribution.</strong> They are included in the patient list but excluded from HCP totals.</p>
<p class="small">__WITHHELD__ attributed priority patients are omitted by the configured minimum count per HCP.</p>
<p class="small">Counts describe identified opportunities. They are not predicted numbers of switches or treatment recommendations. This report contains HCP summaries and no patient identifiers.</p>
</div></section>
<section class="panel"><h2>HCP targeting table</h2>
<div class="controls">
<label>Search HCP / organization<input id="search" type="search" placeholder="Type to filter"></label>
<label>Specialty<select id="specialty"><option value="">All specialties</option></select></label>
<label>Geography<select id="geography"><option value="">All geographies</option></select></label>
<button class="action" id="download">Download current view as CSV</button><span class="count" id="count"></span>
</div><div class="table-wrap"><table><thead><tr>__HEADERS__</tr></thead><tbody id="rows">__ROWS__</tbody></table></div>
<noscript><p>JavaScript is disabled. The full table remains available; use the pipeline's hcp_targets.csv file.</p></noscript>
</section>
<footer><details><summary>Evaluation and provenance</summary><p>__EVALUATION__</p>
<p class="small">Run __RUN__ · All transformations and availability checks run locally. Data and model changes require their own validation.</p></details>
<p class="small">Self-contained report: open this file in a browser. No web server, internet access, external assets or additional visualization packages are required.</p></footer>
</main><script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const allRows=JSON.parse(document.getElementById('data').textContent);
const columns=__COLUMNS__;
let sortKey='hcp_rank',ascending=true,shown=[];
const byId=id=>document.getElementById(id);
['specialty','geography'].forEach(key=>{
 const values=[...new Set(allRows.map(row=>String(row[key]??'')))].sort();
 values.forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=value||'Unknown';byId(key).appendChild(option)});
 byId(key).addEventListener('change',render);
});
byId('search').addEventListener('input',render);
document.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{
 const next=button.dataset.sort;ascending=sortKey===next?!ascending:true;sortKey=next;render();
}));
function render(){
 const search=byId('search').value.toLowerCase();
 shown=allRows.filter(row=>(!byId('specialty').value||String(row.specialty)===byId('specialty').value)&&
 (!byId('geography').value||String(row.geography)===byId('geography').value)&&
 [row.hcp_id,row.organization].join(' ').toLowerCase().includes(search));
 shown.sort((a,b)=>{
  const x=a[sortKey],y=b[sortKey],delta=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y));
  return (ascending?1:-1)*delta;
 });
 const body=byId('rows');body.replaceChildren();
 shown.forEach(row=>{const tr=document.createElement('tr');columns.forEach(key=>{const td=document.createElement('td');td.textContent=String(row[key]??'');tr.appendChild(td)});body.appendChild(tr)});
 byId('count').textContent=shown.length+' HCPs shown';
 const bars=byId('bars');bars.replaceChildren();
 const maximum=shown.reduce((value,row)=>Math.max(value,row.priority_patient_count),1);
 shown.slice(0,12).forEach(row=>{
  const bar=document.createElement('div');bar.className='bar';
  const label=document.createElement('span');label.textContent=row.hcp_id;
  const track=document.createElement('div');track.className='track';
  const fill=document.createElement('div');fill.className='fill';fill.style.width=(100*row.priority_patient_count/maximum)+'%';track.appendChild(fill);
  const count=document.createElement('strong');count.textContent=row.priority_patient_count;
  bar.append(label,track,count);bars.appendChild(bar);
 });
 if(!shown.length){const p=document.createElement('p');p.textContent='No HCPs match this view.';bars.appendChild(p)}
}
function csvCell(value){
 let text=String(value??'');
 if(/^[\\s]*[=+@-]/.test(text))text="'"+text;
 return '"'+text.replaceAll('"','""')+'"';
}
byId('download').addEventListener('click',()=>{
 const keys=__EXPORT_COLUMNS__;
 const lines=[keys.map(csvCell).join(','),...shown.map(row=>keys.map(key=>csvCell(row[key])).join(','))];
 const blob=new Blob(['\\uFEFF'+lines.join('\\r\\n')],{type:'text/csv;charset=utf-8;'});
 const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='hcp_targets_filtered.csv';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></body></html>"""
    replacements = {
        "__FRACTION__": f"{fraction:.0%}",
        "__SCOPE__": escape(manifest["data_scope"]),
        "__DATE__": escape(manifest["scoring_date"]),
        "__EVIDENCE__": escape(evidence_text),
        "__ELIGIBLE__": f"{coverage['eligible_patients']:,}",
        "__PRIORITY__": f"{coverage['priority_patients']:,}",
        "__HCPS__": f"{len(targets):,}",
        "__ATTRIBUTED__": f"{coverage['priority_patients'] - coverage['unattributed_priority_patients']:,}",
        "__UNASSIGNED__": str(coverage["unattributed_priority_patients"]),
        "__WITHHELD__": str(coverage.get("withheld_priority_patients", 0)),
        "__HEADERS__": headings,
        "__ROWS__": rows,
        "__EVALUATION__": escape(evaluation),
        "__RUN__": escape(manifest["run_id"]),
        "__DATA__": payload,
        "__COLUMNS__": json.dumps([key for key, _ in fields]),
        "__EXPORT_COLUMNS__": json.dumps(list(targets.columns)),
    }
    template = re.sub(r"__[A-Z_]+__", lambda match: replacements[match.group()], template)
    Path(path).write_text(template, encoding="utf-8")
