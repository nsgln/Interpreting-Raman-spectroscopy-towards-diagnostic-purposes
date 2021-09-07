conda init
source .bashrc
echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > /home/newuser/.bashrc
source .bashrc
