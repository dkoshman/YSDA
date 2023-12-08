# YSDA
Yandex School of Data Analysis materials 

To set up python environment:
1. scp YSDA/.tmux.conf ~/
2. python3.11 -m venv YSDA/python3.11
3. . YSDA/python3.11/bin/activate
4. pip install -r YSDA/python3.11/requirements
5. jupyter notebook --no-browser --ip 0.0.0.0 --port 6172
6. on local machine:
    ssh -N -f -L localhost:6172:localhost:6172 <user@host>
7. go to localhost:6172 in your browser
