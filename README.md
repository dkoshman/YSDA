# YSDA
Yandex School of Data Analysis materials 

To set up python environment:

0. ssh-keygen -t ed25519
0. eval "$(ssh-agent -s)"
0. ssh-add ~/.ssh/id_ed25519
0. scp YSDA/.tmux.conf ~/
0. Restart tmux
0. python3.11 -m venv YSDA/python3.11
0. alias py311=". YSDA/python3.11/bin/activate"
0. pip install -r YSDA/python3.11/requirements\
   pip install jupyter_contrib_nbextensions\
   python -m jupyter contrib nbextension install --sys-prefix\
   python -m jupyter nbextension enable execute_time/ExecuteTime --sys-prefix\
   python -m jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --sys-prefix
0. alias jn="jupyter notebook --no-browser --ip 0.0.0.0 --port 6172"
0. On local machine:\
    ssh -N -f -L localhost:6172:localhost:6172 [user]@[host]
0. go to localhost:6172 in your browser
0. In python:
```from notebook.services.config import ConfigManager
import pprint

pprint.pprint(
    ConfigManager().update(
        "notebook",
        {
            "ExecuteTime": {
                "display_absolute_timestamps": False,
                "relative_timing_update_period": 5,
                "highlight": {"use": False},
                "template": {"executed": "${duration}", "queued": "queued"},
            }
        },
    )
)
```
