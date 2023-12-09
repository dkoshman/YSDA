# YSDA
Yandex School of Data Analysis materials 

To set up python environment:
0. ssh-keygen -t ed25519
0. eval "$(ssh-agent -s)"
0. ssh-add ~/.ssh/id_ed25519
1. scp YSDA/.tmux.conf ~/
2. Restart tmux
3. python3.11 -m venv YSDA/python3.11
4. alias py311=". YSDA/python3.11/bin/activate"
5. pip install -r YSDA/python3.11/requirements
6. alias jn="jupyter notebook --no-browser --ip 0.0.0.0 --port 6172"
7. On local machine:
    ssh -N -f -L localhost:6172:localhost:6172 <user@host>
8. go to localhost:6172 in your browser
