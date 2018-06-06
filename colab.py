'''
Copy and paste this into a new notebook in Google Colab:
https://colab.research.google.com/

To train on GPUs, go to Edit -> Notebook Settings and set
"Hardware accelerator" to "GPU".
'''

import os, shutil

! apt install gdal-bin python-gdal python3-gdal
! pwd
! ls -alh

# upload the file to Google Drive and examine the download url to find the id
gid_by_fname = {
    'Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf': '1a6MeRs3ep5aNF7ZN0Bu4YDNvRroddlQ7',
    'snlworkbook_Combined_fixed.xls': '1mqQeZwKq-JLvGpy0P3voHnASdhLURV5I'
}
dir_by_fname = {
    'Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf': 'data/d950396',
    'snlworkbook_Combined_fixed.xls': 'data'
}
for fname, gid in gid_by_fname.items():
  dir = dir_by_fname[fname]
  try:
    open(os.path.join(dir, fname))
  except Exception as exc:
    print('exc: %s' % exc)
    if not os.path.exists('./gdrive'):
      print('Downloading gdrive executable...')
      ! wget "https://docs.google.com/uc?id=0B3X9GlR6EmbnQ0FtZmJJUXEyRTA&export=download" -O gdrive
      ! chmod +x gdrive
    print('Downloading %s (%s)' % (fname, gid))
    # get the SSO code by signing in at the link printed to stdout:
    # https://accounts.google.com/o/oauth2/auth?...
    ! echo "<sso_code>" | ./gdrive download {gid}
    ! mkdir -p {dir}
    ! mv '{fname}' {dir}

urls_by_modulename = {
    'percache': 'git+https://github.com/abrichr/percache.git@default_args',
    'pexpect': 'pexpect',
    'hipsterplot': 'hipsterplot',
    'affine': 'affine',
    'mapkit': 'mapkit',
    'pyproj': 'pyproj',
    'utm': 'utm',
    'xlrd': 'xlrd'
}

for modulename, url in urls_by_modulename.items():
  try:
    __import__(modulename)
  except:
    ! pip install {url}

try:
  open('/root/.ssh/colab')
except:
  from google.colab import files
  '''
  Create a tar.gz file containing files for password-less ssh:
    mkdir -p ~/colab/ssh
    ssh-keygen -t rsa -N '' -f ~/colab/ssh/colab
    echo -e "Host bitbucket.org\n  Hostname bitbucket.org\n  IdentityFile ~/.ssh/colab" > ~/colab/ssh/config
    tar -zcvf ~/colab/ssh.tar.gz ~/colab/ssh
  Select ssh.tar.gz after clicking on the the "Choose files" button
  (you may need to re-run the script to see it)
  '''
  uploaded = files.upload()
  ! rm -rf /root/.ssh
  ! mkdir /root/.ssh
  ! tar -xvzf ssh.tar.gz
  ! cp ssh/* /root/.ssh && rm -rf ssh && rm -rf ssh.tar.gz
  ! chmod 700 /root/.ssh
  ! ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
  ! chmod 644 /root/.ssh/known_hosts

! rm -rf minalytics
! ls -alh
! git clone git@bitbucket.org:abrichr/minalytics.git
! ls -alh minalytics

from minalytics import prospector
prospector.hyperopt()
