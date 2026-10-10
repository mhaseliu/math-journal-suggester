# Security

Do not post credentials or unpublished manuscript text in issues. Report suspected vulnerabilities privately through GitHub's security advisory reporting when enabled.

The local website binds only to loopback. arXiv import accepts restricted IDs/official abstract links, uses fixed endpoints, blocks redirects and checks identity, size and mathematics categories. There is no PDF parser or arbitrary URL fetch. Public hosting requires an isolated model worker, metadata worker and bounded gateway. Publishing code does not configure those services.

Only load model checkpoints you trust and verify their hashes. No secrets or manuscript data are required for the CPU test suite.
