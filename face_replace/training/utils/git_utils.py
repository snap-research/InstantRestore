import os
from typing import Optional

import git


def get_git_info() -> Optional[str]:
    r = git.Repo(os.getcwd())
    res = []
    try:
        _ = r.git_dir
        sha = r.head.object.hexsha
        res.append(f"COMMIT ID: {sha} (in branch {r.active_branch.name}")
        res.append('')
        res.append("UNCOMMITED CHANGES:")
        for x in r.index.diff(None):
            if ".ipynb" in x.a_path:  # Don't want to log ipynb files
                continue
            res.append(f"\n------- FOR FILE {x.a_path} -------")
            diff = r.git.diff(x.a_path)
            res.append(diff)
    except Exception:
        return None
    return "\n".join(res)
