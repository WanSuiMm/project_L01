# Round 9 attachment snapshot

The accompanying proof preserves the mathematical text of the user-supplied
Round 9 ZIP; two trailing line-end spaces are normalized for publication.
It is not the completed experiment on the original Round 6 models. The
unchanged source ZIP is retained locally with the hash below.

The attachment's independent width-16 trained fixture and reported numerical
examples differ from the original width-64 Round 6 checkpoints. Its claimed
512-episode quality evaluation and 18 extra seed checks were not accompanied
by reproducible procedures/logs. Those claims are retained as source text,
not promoted to verified evidence here.

The prospective original-model experiment is specified in
[the frozen protocol](../../../experiments/PROTOCOL_ROUND9_LOCAL_CERT_V1.md).
The operational verifier derives from the attachment, changes checkpoint
loading to weights-only, and adds final LayerNorm to tube statistics. The
reported local bound values are otherwise unchanged on the replayed fixture.
All results retain the float64, non-interval qualification.

Source ZIP SHA256:
`9b54ad3dbf5e2ffd0e05223bef90c1419c0425bbfbf51d4b2f2fc18fb94da5f7`.
