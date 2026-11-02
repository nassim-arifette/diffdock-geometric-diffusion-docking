"""Portable HTML inspection of stored poses, optional evaluation and actual sampler frames.

Molecular data is embedded; the default renderer is pinned 3Dmol.js from a CDN.
Pass ``three_dmol_js`` to embed a local copy and make the document fully offline.
No reference-based ranking or molecular evaluation is performed by this viewer.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from ._util import read_json, sha256_file
from .manifest import MANIFEST_NAME, load_manifest, referenced_files, verify_manifest

THREE_DMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"


class InspectorError(ValueError):
    """Stored data cannot be associated with this run without ambiguity."""


def _local_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise InspectorError(f"file is outside the run directory: {relative}") from exc
    return path


def _read_entry(run_dir: Path, entry: dict) -> str:
    path = _local_path(run_dir, entry["path"])
    if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
        raise InspectorError(f"hash mismatch: {entry['path']}")
    return path.read_text(encoding="utf-8", errors="replace")


def _unique_by(items: list, key: str, label: str) -> dict:
    result = {}
    for item in items:
        value = item[key]
        if value in result:
            raise InspectorError(f"duplicate {label}: {value}")
        result[value] = item
    return result


def _trajectory_data(run_dir: Path, pose: dict) -> dict | None:
    if not pose.get("frames"):
        return None
    from . import frames

    try:
        from rdkit import Chem

        sdf = _local_path(run_dir, pose["path"])
        traj = frames.trajectory(_local_path(run_dir, pose["frames"]["path"]), sdf)
        mol = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)[0]
        return {
            "pdb": frames.to_multimodel_pdb(mol, traj),
            "frames": [{"kind": f.kind, "step": f.step} for f in traj.frames],
            "notes": traj.notes,
            "n_models_in_file": traj.n_models_in_file,
        }
    except (ImportError, frames.FrameError, ValueError, RuntimeError) as exc:
        return {"pdb": None, "frames": [], "notes": [f"Trajectory unavailable: {exc}"]}


def build_inspector_data(run: Path | str, *, evaluation: Path | str | None = None) -> dict:
    """Read verified run files; attach scores only when their recorded provenance matches."""
    run_path = Path(run).resolve()
    run_dir = run_path if run_path.is_dir() else run_path.parent
    manifest = load_manifest(run_path)
    manifest_path = run_dir / MANIFEST_NAME
    for entry in referenced_files(manifest):
        _local_path(run_dir, entry["path"])
    verify_manifest(run_dir, manifest)
    manifest_hash = sha256_file(manifest_path)
    eval_path = Path(evaluation) if evaluation is not None else run_dir / "evaluation.json"
    report = None
    if evaluation is not None or eval_path.is_file():
        report = read_json(eval_path)
        if report.get("schema_version") != 1:
            raise InspectorError("unsupported evaluation schema")
        if report.get("manifest_sha256") != manifest_hash:
            raise InspectorError("evaluation manifest_sha256 does not match this manifest")
        if report.get("run_id") != manifest.get("run_id"):
            raise InspectorError("evaluation run_id does not match this run")
    evaluated = _unique_by((report or {}).get("complexes", []), "complex_name", "evaluated complex")
    complexes = _unique_by(manifest.get("complexes", []), "complex_name", "complex")
    if set(evaluated) - set(complexes):
        raise InspectorError("evaluation contains complexes absent from the manifest")
    data = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "error": manifest.get("error"),
        "seed": manifest.get("seed"),
        "manifest_sha256": manifest_hash,
        "upstream": manifest.get("upstream"),
        "inference_parameters": manifest.get("inference_parameters"),
        "evaluation_status": (report or {}).get("status"),
        "protocol": (report or {}).get("protocol"),
        "backends": (report or {}).get("backends"),
        "complexes": [],
    }
    for name, complex_entry in complexes.items():
        ec = evaluated.get(name, {})
        poses = _unique_by(complex_entry.get("poses", []), "pose_id", "pose")
        metrics = _unique_by(ec.get("poses", []), "pose_id", "evaluated pose")
        if set(metrics) - set(poses):
            raise InspectorError(f"evaluation contains unknown poses for {name}")
        c = {
            "name": name,
            "status": complex_entry.get("status"),
            "error": complex_entry.get("error"),
            "receptor_pdb": _read_entry(run_dir, complex_entry["receptor"]),
            "reference_sdf": None,
            "reference_note": None,
            "poses": [],
            "variants": ec.get("variants", {}),
            "oracle": ec.get("oracle", {}),
        }
        reference = complex_entry.get("reference")
        if reference:
            if Path(reference["path"]).suffix.lower() not in {".sdf", ".mol"}:
                c["reference_note"] = "Reference overlay supports SDF/MOL files only."
            else:
                c["reference_sdf"] = _read_entry(run_dir, reference)
                c["reference_note"] = reference.get("provenance")
        ids = set(poses)
        for pose in sorted(poses.values(), key=lambda p: (p["rank"], p["pose_id"])):
            metric = metrics.get(pose["pose_id"])
            if metric and (metric.get("sha256") != pose.get("sha256") or metric.get("path") != pose["path"]):
                raise InspectorError(f"evaluation pose provenance mismatch: {pose['pose_id']}")
            c["poses"].append({
                "pose_id": pose["pose_id"], "parent_pose_id": None,
                "rank": pose["rank"], "kind": "raw", "path": pose["path"],
                "confidence": pose.get("confidence"),
                "confidence_rounded": pose.get("confidence_rounded"),
                "sdf": _read_entry(run_dir, pose), "metrics": metric,
                "trajectory": _trajectory_data(run_dir, pose),
            })
            refined = (metric or {}).get("refined")
            if refined and refined.get("path"):
                rid = refined["pose_id"]
                if rid in ids:
                    raise InspectorError(f"duplicate refined pose identifier: {rid}")
                ids.add(rid)
                if not refined.get("sha256"):
                    raise InspectorError(f"refined pose has no sha256: {rid}")
                c["poses"].append({
                    "pose_id": rid, "parent_pose_id": pose["pose_id"],
                    "rank": pose["rank"], "kind": "refined", "path": refined["path"],
                    "confidence": None, "parent_confidence": pose.get("confidence"),
                    "confidence_rounded": None,
                    "sdf": _read_entry(run_dir, refined), "metrics": refined,
                    "trajectory": None,
                })
        for variant in c["variants"].values():
            if set(variant.get("selected_pose_ids", [])) - ids:
                raise InspectorError(f"variant refers to an unknown pose in {name}")
        data["complexes"].append(c)
    return data


def _json_for_html(data: dict) -> str:
    # JSON remains data even when a ligand title contains HTML or a closing script tag.
    return (json.dumps(data, ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_inspector(
    run: Path | str,
    output: Path | str | None = None,
    *,
    evaluation: Path | str | None = None,
    three_dmol_js: Path | str | None = None,
) -> Path:
    """Write a shareable HTML document, defaulting to ``<run>/inspector.html``.

    ``evaluation=None`` discovers ``<run>/evaluation.json`` if present. A missing
    report leaves measurements explicitly unevaluated. The function never docks,
    scores, downloads code or sends molecular data to a service.
    """
    data = build_inspector_data(run, evaluation=evaluation)
    run_path = Path(run).resolve()
    run_dir = run_path if run_path.is_dir() else run_path.parent
    target = Path(output) if output is not None else run_dir / "inspector.html"
    title = f"DiffDock · {data['run_id']}"
    if three_dmol_js is None:
        renderer = f'<script src="{THREE_DMOL_URL}"></script>'
        mode = "Molecular data embedded locally. Rendering requires the 3Dmol.js CDN and WebGL."
    else:
        payload = base64.b64encode(Path(three_dmol_js).read_bytes()).decode("ascii")
        renderer = f'<script src="data:text/javascript;base64,{payload}"></script>'
        mode = "Molecular data and renderer embedded. This document can be viewed offline with WebGL."
    page = (_PAGE.replace("__TITLE__", html.escape(title))
            .replace("__MODE__", html.escape(mode))
            .replace("__RENDERER__", renderer)
            .replace("__DATA__", _json_for_html(data)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return target


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#172b3a;background:#f3f5f7}
*{box-sizing:border-box}body{max-width:1440px;margin:0 auto;padding:24px}h1{font-size:25px;margin:0}h2{font-size:18px;margin:0 0 12px}p{margin:8px 0}.muted{color:#526370}.small{font-size:13px}
header{margin-bottom:20px}section{background:white;border:1px solid #d4dce2;border-radius:8px;padding:18px;margin-bottom:16px}.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(280px,1fr);gap:16px}
label{display:inline-flex;align-items:center;gap:6px}select,button{font:inherit;padding:7px 10px;background:white;border:1px solid #a7b7c2;border-radius:4px;color:inherit}button{cursor:pointer}button:hover{background:#edf4f8}button:disabled{cursor:default;opacity:.5}select{max-width:100%}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
#viewer{height:510px;position:relative;background:#f8fafb;border:1px solid #dae2e8;border-radius:4px}#viewer-error{padding:16px;color:#9a3412}.legend{display:flex;gap:18px;font-size:13px;flex-wrap:wrap;margin:10px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.pred{background:#00a5b5}.ref{background:#cb4eb9}.rec{background:#9da8b2}
dl{display:grid;grid-template-columns:1fr 1.2fr;gap:8px;margin:0}dt{color:#526370}dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.pass{color:#176b3d}.fail{color:#ad3324}.unknown{color:#526370}.notice{border-left:3px solid #d3a034;padding:5px 12px;background:#fff9e9}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e1e7eb;vertical-align:top}th{color:#526370;font-weight:600}tbody tr.selected{background:#e7f4f7}code,pre{font-family:ui-monospace,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}input[type=range]{flex:1;min-width:120px}details{margin-top:12px}summary{cursor:pointer}#frame-label{min-width:130px}.frame-bar{display:flex;align-items:center;gap:12px}#checks td:first-child{overflow-wrap:anywhere}footer{padding:0 4px 20px}.error{white-space:pre-wrap;color:#ad3324}
@media(max-width:800px){body{padding:12px}.layout{display:block}#viewer{height:400px}dl{grid-template-columns:1fr 1fr}.controls label{max-width:100%}}
</style></head><body>
<header><h1>DiffDock pose inspector</h1><p id="run-info" class="muted"></p><p class="small muted">__MODE__</p></header>
<section><div class="controls"><label>Complex <select id="complex"></select></label><label>Pose <select id="pose"></select></label></div><p id="run-error" class="error" role="status"></p>
<p class="small muted">Confidence ranks poses; it is not binding affinity. Refined poses keep a link to their original candidate.</p></section>
<div class="layout"><div><section>
<div class="controls"><label><input id="show-reference" type="checkbox">Reference overlay (diagnostic)</label><label><input id="show-pocket" type="checkbox" checked>Nearby side chains (5 Å)</label><button id="focus">Focus pose</button><button id="fit">Fit receptor</button></div>
<div id="viewer"><div id="viewer-error" role="status">Loading molecular renderer…</div></div>
<div class="legend"><span><i class="dot pred"></i>Prediction</span><span><i class="dot ref"></i>Reference</span><span><i class="dot rec"></i>Receptor</span></div>
<p id="reference-note" class="small muted"></p>
<div id="trajectory-panel"><h2>Recorded reverse diffusion</h2><div class="frame-bar"><button id="play" disabled>Play</button><input id="frame" type="range" min="0" max="0" value="0" disabled aria-label="Recorded diffusion frame"><span id="frame-label" class="small"></span></div><p id="trajectory-notes" class="small muted"></p></div>
</section><section><h2>Candidate pool</h2><div class="scroll"><table><thead><tr><th>Candidate</th><th>Kind</th><th>Confidence</th><th>RMSD (Å)</th><th>Valid</th><th>CNNscore</th></tr></thead><tbody id="poses"></tbody></table></div></section>
<section><h2>Selection variants</h2><p class="small muted">These selections come from the evaluation report. Reference RMSD is a diagnostic only.</p><div class="scroll"><table><thead><tr><th>Variant</th><th>Selected top-1</th><th>Joint success</th></tr></thead><tbody id="variants"></tbody></table></div><p id="oracle" class="small muted"></p></section></div>
<div><section><h2>Selected pose</h2><dl id="metrics"></dl><p id="pose-error" class="error small"></p><p class="small notice">RMSD and validity below describe the final stored pose, including while you browse earlier diffusion frames. No alignment is applied by this viewer.</p></section>
<section><h2>Validity checks</h2><p id="evaluation-note" class="small muted"></p><table><tbody id="checks"></tbody></table></section>
<section><h2>Provenance</h2><details><summary>Protocol and software</summary><pre id="provenance"></pre></details></section></div></div>
<footer class="small muted">Rotate: drag · zoom: wheel/pinch. The reference is shown in its stored receptor frame; an overlay alone does not establish matching chemical identity.</footer>
<script id="inspector-data" type="application/json">__DATA__</script>
__RENDERER__
<script>
"use strict";
const data=JSON.parse(document.getElementById("inspector-data").textContent);
const el=id=>document.getElementById(id);
const missing="Not evaluated";
const value=x=>x===null||x===undefined?missing:String(x);
const bool=x=>x===true?"Pass":x===false?"Fail":missing;
const setText=(id,text)=>{el(id).textContent=text};
const td=(row,text)=>{const cell=document.createElement("td");cell.textContent=text;row.appendChild(cell);return cell};
const option=(select,text,v)=>{const o=document.createElement("option");o.textContent=text;o.value=v;select.appendChild(o)};
let current=null, selected=null, viewer=null, ligandModel=null, timer=null, drawVersion=0;
function stop(){if(timer!==null){clearInterval(timer);timer=null;}setText("play","Play");}
function label(p){return "Rank "+p.rank+" · "+p.kind+" · "+p.pose_id;}
function appendMetric(name,val){const a=document.createElement("dt"),b=document.createElement("dd");a.textContent=name;b.textContent=val;el("metrics").append(a,b);}
function drawTable(){
 el("poses").replaceChildren();
 for(const p of current.poses){const row=document.createElement("tr");if(p===selected)row.className="selected";
 const cell=td(row,"");const button=document.createElement("button");button.textContent="Rank "+p.rank;button.title=p.pose_id;button.onclick=()=>choose(p.pose_id);cell.appendChild(button);
 const m=p.metrics||{};td(row,p.kind);td(row,p.kind==="refined"?"—":value(p.confidence));td(row,value(m.rmsd_angstrom));td(row,bool(m.valid));td(row,value(m.gnina_score));el("poses").appendChild(row);}
}
function variantTable(){
 el("variants").replaceChildren();const variants=Object.entries(current.variants);
 if(!variants.length){const row=document.createElement("tr");td(row,missing);el("variants").appendChild(row);}
 for(const [name,v] of variants){const row=document.createElement("tr");td(row,name);const cell=td(row,"");const id=(v.selected_pose_ids||[])[0];
 if(id){const b=document.createElement("button");b.textContent=id;b.onclick=()=>choose(id);cell.appendChild(b);}else cell.textContent=v.available===false?"Unavailable":v.empty_selection?"Abstention":missing;
 td(row,bool(v.top1_joint_success));el("variants").appendChild(row);}
 const o=current.oracle;setText("oracle",Object.keys(o).length?"Oracle diagnostics (not a selectable method): "+Object.entries(o).map(([k,v])=>k+" = "+value(v)).join(" · "):"Oracle: "+missing);
}
function details(){
 const p=selected,m=p.metrics||{};el("metrics").replaceChildren();
 appendMetric("Pose ID",p.pose_id);appendMetric("Initial rank",String(p.rank));appendMetric("Coordinates",p.kind==="refined"?"Locally refined":"Raw DiffDock");
 appendMetric("Confidence (full precision)",p.kind==="refined"?"Not recomputed":value(p.confidence));
 if(p.kind==="refined"){appendMetric("Parent pose",p.parent_pose_id);appendMetric("Parent confidence",value(p.parent_confidence));}
 else if(p.confidence===null)appendMetric("Rounded filename score",value(p.confidence_rounded));
 appendMetric("RMSD (Å)",value(m.rmsd_angstrom));appendMetric("Physical validity",bool(m.valid));appendMetric("GNINA CNNscore",value(m.gnina_score));appendMetric("Stored file",p.path);
 setText("pose-error",(m.errors||[]).map(value).join("\n"));el("checks").replaceChildren();
 for(const [name,result] of Object.entries(m.checks||{})){const row=document.createElement("tr");td(row,name);const cell=td(row,bool(result));cell.className=result===true?"pass":result===false?"fail":"unknown";el("checks").appendChild(row);}
 setText("evaluation-note",p.metrics?"Recorded checks; missing values remain unevaluated.":"Not evaluated: no evaluation record for this pose.");
 const tr=p.trajectory,hasFrames=!!(tr&&tr.pdb&&tr.frames.length);el("frame").disabled=!hasFrames;el("play").disabled=!hasFrames;
 el("frame").max=hasFrames?tr.frames.length-1:0;el("frame").value=el("frame").max;
 setText("frame-label",hasFrames?"Final pose":"No saved frames");
 setText("trajectory-notes",tr?(tr.notes||[]).join(" "):p.kind==="refined"?"No diffusion trajectory is attributed to the refined coordinates. Select the parent candidate to inspect its recorded trajectory.":"This run did not store a trajectory for this pose.");
}
async function draw(reset=true){
 if(!viewer||!selected)return;
 const token=++drawVersion,p=selected,tr=p.trajectory,hasFrames=!!(tr&&tr.pdb&&tr.frames.length);
 const old=viewer.getView();viewer.removeAllModels();
 viewer.addModel(current.receptor_pdb,"pdb");
 viewer.setStyle({model:0},{cartoon:{color:"#aab4bc",opacity:.85}});
 ligandModel=hasFrames?viewer.addModelsAsFrames(tr.pdb,"pdb"):viewer.addModel(p.sdf,"sdf");
 if(hasFrames)await ligandModel.setFrame(Number(el("frame").value));
 if(token!==drawVersion)return;
 viewer.setStyle({model:1},{stick:{colorscheme:"cyanCarbon",radius:.2}});
 if(el("show-pocket").checked)viewer.addStyle({model:0,within:{distance:5,sel:{model:1}}},{stick:{colorscheme:"grayCarbon",radius:.12}});
 // addModels() returns the viewer's whole model list in the pinned renderer.
 // Add reference records individually so styling cannot recolor the prediction/receptor.
 if(el("show-reference").checked&&current.reference_sdf){for(const block of current.reference_sdf.split(/^\$\$\$\$[ \t]*\r?$/m)){if(block.trim())viewer.addModel(block+"\n$$$$\n","sdf").setStyle({},{stick:{colorscheme:"magentaCarbon",radius:.15}});}}
 if(reset){viewer.zoomTo({model:1});viewer.zoom(.8);}else viewer.setView(old);viewer.render();
}
async function frameChanged(){
 if(!selected||!selected.trajectory||!selected.trajectory.frames.length)return;
 const i=Number(el("frame").value),f=selected.trajectory.frames[i];
 setText("frame-label",f.kind==="final"?"Final pose":f.kind==="random_start"?"Random start":"Denoising step "+f.step);
 await draw(false);
}
function choose(id){stop();selected=current.poses.find(p=>p.pose_id===id)||null;el("pose").value=id;drawTable();if(selected){details();draw();}else{++drawVersion;el("metrics").replaceChildren();el("checks").replaceChildren();el("frame").disabled=true;el("play").disabled=true;setText("frame-label","No saved frames");setText("trajectory-notes","No stored pose in this complex.");setText("pose-error","");if(viewer){viewer.removeAllModels();viewer.addModel(current.receptor_pdb,"pdb");viewer.setStyle({cartoon:{color:"#aab4bc"}});viewer.zoomTo().render();}setText("evaluation-note","No stored poses in this complex.");}}
function changeComplex(){
 stop();current=data.complexes[Number(el("complex").value)];el("pose").replaceChildren();
 if(!current)return;for(const p of current.poses)option(el("pose"),label(p),p.pose_id);
 el("show-reference").disabled=!current.reference_sdf;el("show-reference").checked=false;
 setText("reference-note",current.reference_sdf?"Reference overlay is diagnostic only. All stored reference records are displayed without alignment. "+(current.reference_note||""):current.reference_note||"No reference ligand supplied; reference-based measurements may be unavailable.");
 setText("run-error",current.error||data.error||"");variantTable();choose(current.poses.length?current.poses[0].pose_id:"");
}
setText("run-info",value(data.run_id)+" · status "+value(data.status)+" · seed "+value(data.seed)+" · evaluation "+value(data.evaluation_status));
setText("provenance",JSON.stringify({manifest_sha256:data.manifest_sha256,upstream:data.upstream,inference_parameters:data.inference_parameters,protocol:data.protocol,backends:data.backends},null,2));
for(let i=0;i<data.complexes.length;i++)option(el("complex"),data.complexes[i].name,i);
try{if(!window.$3Dmol)throw new Error("3Dmol.js unavailable. Connect to its CDN or regenerate with an embedded local renderer. Tables remain available.");el("viewer-error").remove();viewer=$3Dmol.createViewer(el("viewer"),{backgroundColor:"#f8fafb"});}catch(error){let box=el("viewer-error");if(!box){box=document.createElement("div");box.id="viewer-error";el("viewer").appendChild(box);}box.textContent=error.message;}
el("complex").onchange=changeComplex;el("pose").onchange=()=>choose(el("pose").value);
el("show-reference").onchange=()=>draw(false);el("show-pocket").onchange=()=>draw(false);
el("focus").onclick=()=>{if(viewer&&selected)viewer.zoomTo({model:1}).zoom(.8).render()};el("fit").onclick=()=>{if(viewer)viewer.zoomTo().render()};
el("frame").oninput=()=>{stop();frameChanged()};
el("play").onclick=()=>{if(timer!==null){stop();return;}if(!selected||!selected.trajectory)return;el("frame").value=0;frameChanged();setText("play","Pause");timer=setInterval(()=>{if(Number(el("frame").value)>=Number(el("frame").max)){stop();return;}el("frame").value=Number(el("frame").value)+1;frameChanged();},350)};
window.addEventListener("resize",()=>{if(viewer)viewer.resize().render()});
changeComplex();
</script></body></html>'''


def main(argv=None):
    """Export an inspectable run using the same checks as the notebook API."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help=f"directory containing {MANIFEST_NAME}")
    parser.add_argument("--out", type=Path, help="output HTML (default: run/inspector.html)")
    parser.add_argument("--evaluation", type=Path, help="saved evaluation JSON for this run")
    parser.add_argument("--three-dmol-js", type=Path, help="embed a local 3Dmol renderer for offline use")
    args = parser.parse_args(argv)
    print(write_inspector(args.run, args.out, evaluation=args.evaluation, three_dmol_js=args.three_dmol_js))


if __name__ == "__main__":
    main()
