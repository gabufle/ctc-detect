
## Git rules — all agents must follow

ALWAYS follow this exact git workflow at the end of every task:
1. git add -A
2. Check for large files: git status (verify nothing over 50MB is staged)
3. git commit -m "descriptive message"
4. git pull --rebase origin main
5. git push origin main

NEVER commit these files — too large for GitHub:
- data/processed/tokenized*/
- data/processed/*.h5ad
- data/raw/
- results/checkpoints/
- Geneformer/
- *.arrow
- *.bin
- *.safetensors
- *.pt
- *.pkl (unless tiny config files)

If push fails due to large files already committed:
  git rm -r --cached <large_file_or_dir>
  echo "<pattern>" >> .gitignore
  git add .gitignore
  git commit -m "remove large files from tracking"
  git pull --rebase origin main
  git push origin main

Never leave work uncommitted or unpushed at end of task.
If rebase fails due to conflicts, block the task and report.
