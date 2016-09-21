# mne_tools
Tools for python MNE Toolbox for EEG data analysis
## read_antcnt
The libeep toolbox is from ANT, you can find the source and try to compile yourself [here](https://sourceforge.net/projects/libeep/)
### Install
- Install MNE-toolbox
- Copy the content of libeep into the .?/python2.7/site-packages/libeep/ folder
- add read_antcnt to your path

### How-To:
``` python
import read_antcnt
raw = read_antcnt('filename.cnt')
raw.plot()
```
### Custom NBP-Things
In order to set the AUX/BIP channels as 'misc' use the following code:

``` python
raw.set_channel_types({ch:'misc' for ch in raw.ch_names if (ch.find('AUX')==0) | (ch.find('BIP')==0)}) # 
```


### Thanks
Thanks to Robert Smies for providing me with a pre-compiled version of libeep




