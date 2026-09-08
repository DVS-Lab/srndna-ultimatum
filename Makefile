.PHONY: test reviewer-behavior reviewer-imaging-audit

test:
	bash code/validate_workflow.sh

reviewer-behavior:
	bash code/run_logged.sh reviewer-task-events python3 code/audit_task_events.py
	bash code/run_logged.sh reviewer-behavior Rscript code/analyze_reviewer_behavior.R

reviewer-imaging-audit:
	bash code/run_logged.sh reviewer-image-headers python3 code/audit_image_headers.py
	bash code/run_logged.sh reviewer-l3-template-inputs python3 code/audit_l3_template_inputs.py
	bash code/run_logged.sh reviewer-l3-designs Rscript code/audit_l3_designs.R
	bash code/run_logged.sh reviewer-roi-influence Rscript code/audit_roi_influence.R
