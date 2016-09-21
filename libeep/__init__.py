import pyeep
###############################################################################
class cnt_base:
  def __init__(self, handle):
    if handle == -1:
      raise Exception('not a valid libeep handle')
    self._handle = handle

  def __del__(self):
    pyeep.close(self._handle)
###############################################################################
class cnt_in(cnt_base):
  def __init(self, handle):
    cnt_base.__init__(self, handle)

  def get_channel_count(self):
    return pyeep.get_channel_count(self._handle)

  def get_channel(self, index):
    return (pyeep.get_channel_label(self._handle, index), pyeep.get_channel_unit(self._handle, index), pyeep.get_channel_reference(self._handle, index))

  def get_sample_frequency(self):
    return pyeep.get_sample_frequency(self._handle)

  def get_sample_count(self):
    return pyeep.get_sample_count(self._handle)

  def get_samples(self, fro, to):
    return pyeep.get_samples(self._handle, fro, to)

  def get_trigger_count(self):
    return pyeep.get_trigger_count(self._handle)

  def get_trigger(self, index):
    return pyeep.get_trigger(self._handle, index)
###############################################################################
def read_cnt(filename):
  if not filename.endswith('.cnt'):
    raise Exception('unsupported extension')
  return cnt_in(pyeep.read(filename))
