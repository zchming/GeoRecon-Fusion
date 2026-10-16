"""Fixed-priority, evidence-gated whole-scene choice. No GT or file reads."""
from ..data.protocol import validate_result
METHODS={'colmap','vggt_raw','vggt_ba'}


def validate_policy(policy):
    if policy['schema_version']!=1 or len(policy['priority'])!=3 or set(policy['priority'])!=METHODS:
        raise ValueError('policy must contain each of the three baseline methods exactly once')
    if policy['gate']!='complete_cameras_and_nonempty_evidence' or policy['partial_evidence']!='allow_with_warning' or policy['no_candidate']!='abstain':
        raise ValueError('unsupported selection gate; changing policy family requires a new protocol')
    return policy


def choose(candidates,diagnostics,policy,scene_id,manifest_hash,image_ids,prediction_hashes,availability_errors=None):
    validate_policy(policy)
    if diagnostics['scene_id']!=scene_id or diagnostics['input_manifest_sha256']!=manifest_hash:
        raise ValueError('diagnostics input identity differs from the requested scene')
    trace=[]
    selected=None
    for method in policy['priority']:
        prediction=candidates.get(method);diag=diagnostics['methods'].get(method);reasons=list((availability_errors or {}).get(method,[]));valid=False
        if prediction is None:reasons.append('prediction_missing')
        else:
            try:
                validate_result(prediction);valid=True
            except (ValueError,KeyError,TypeError):reasons.append('invalid_prediction_protocol')
            if valid and prediction.get('role')!='prediction':reasons.append('not_a_method_prediction')
            if valid and (prediction.get('scene_id')!=scene_id or prediction.get('input_manifest_sha256')!=manifest_hash or [r.get('image_id') for r in prediction['images']]!=image_ids):
                reasons.append('input_identity_mismatch')
            if valid and (prediction['status']!='success' or len(prediction['images'])!=len(image_ids) or not all(r['registered'] for r in prediction['images'])):
                reasons.append('no_complete_successful_camera_output')
        if diag is None:reasons.append('diagnostics_missing')
        else:
            if diag.get('prediction_sha256')!=prediction_hashes.get(method) or not prediction_hashes.get(method):reasons.append('stale_or_mismatched_diagnostics')
            if valid and diag.get('baseline_status')!=prediction.get('status'):reasons.append('diagnostic_baseline_status_mismatch')
            count=diag.get('summary',{}).get('evidence_pairs',0)
            if type(count) is not int or count<=0 or diag.get('status') not in {'evidence_available','partial_evidence'}:
                reasons.append('no_usable_geometric_evidence')
            elif count>len(image_ids)*(len(image_ids)-1)//2:reasons.append('invalid_evidence_count')
        eligible=not reasons
        trace.append({'method':method,'eligible':eligible,'reasons':reasons,'evidence_status':diag.get('status') if diag else None})
        if selected is None and eligible:selected=method
    warning='partial_evidence_not_whole_scene_correctness' if selected and diagnostics['methods'][selected]['status']=='partial_evidence' else None
    return {'outcome':'selected' if selected else 'abstain','selected_method':selected,'warning':warning,'trace':trace,
            'basis':'fixed dev-fitted priority among eligible whole-scene outputs; not minimum residual or calibrated probability',
            'fallback_used':bool(selected and selected!=policy['priority'][0]),'correctness_probability':None}
