"""Wayback Machine historical-evidence pipeline.

A self-contained sub-project that reconstructs the March-2023 (GPT-4 launch)
homepages of our classified startups from the Internet Archive and turns them
into a ``classifier_input_2023.csv``. Its evidence-recovery pipeline stays
independent of the production classifier. The shared cleaner is vendored in
:mod:`wayback_machine.evidence` and guarded by a golden test.
:mod:`wayback_machine.classify_2023` exits immediately. Production
classification is ``python -m two_pass_classifier``.

Run order: see README.md.
"""
