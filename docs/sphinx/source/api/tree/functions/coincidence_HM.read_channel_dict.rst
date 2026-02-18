coincidence_HM.read_channel_dict
================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Reads off boolean channel dict entries for times

Signature
---------

.. code-block:: python

   def read_channel_dict(trig_obj, times, chan_name)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trig_obj``
     - -
     - -
     - Instance of trig.TriggerList
   * - ``times``
     - -
     - -
     - Time or list of times to read the channel dict for
   * - ``chan_name``
     - -
     - -
     - Key into trig_obj.channel_dict

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - channel dict entry or entries corresponding to times

Docstring
---------

.. code-block:: text

   Reads off boolean channel dict entries for times
   :param trig_obj: Instance of trig.TriggerList
   :param times: Time or list of times to read the channel dict for
   :param chan_name: Key into trig_obj.channel_dict
   :return: channel dict entry or entries corresponding to times
