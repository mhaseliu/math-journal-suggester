# Publish the prepared adapter on Hugging Face

The model package has been prepared separately from the Git repository. It is not uploaded yet. Use the verified `kev-math-epoch2` directory containing the model card, license and `SHA256SUMS`.

```bash
hf auth login
hf upload YOUR_ACCOUNT/kev-math-journal-suggester /path/to/kev-math-epoch2 . --repo-type model
```

Keep tokens out of chat and Git. After uploading, pin the Hub commit in the code repository and replace the pending-availability note with the actual model link. Do not upload the collected reference corpus or private experiment archive.

[Official CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
