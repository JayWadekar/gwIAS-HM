triggers_single_detector_HM.TriggerList.save_candidatelist
==========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def save_candidatelist(self, trig_fname, append = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trig_fname``
     - -
     - -
     - Absolute path to trigger file to save triggers to
   * - ``append``
     - -
     - True
     - Flag indicating whether to append to existing file If append is True, 1. If trig_fname doesn't exist, we create it and save self.processedclist to it 2. If trig_fname exists, we append self.processedclist to its contents

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   :param trig_fname: Absolute path to trigger file to save triggers to
   :param append: Flag indicating whether to append to existing file
   If append is True,
       1. If trig_fname doesn't exist, we create it and save
          self.processedclist to it
       2. If trig_fname exists, we append self.processedclist to its
          contents
   :return Updates self.trig_fname
