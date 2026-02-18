coincidence_HM.find_friends_in_cloud
====================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def find_friends_in_cloud(cloud_dict, finer_calphas)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cloud_dict``
     - -
     - -
     - Dictionary with processedclists indexed by template id
   * - ``finer_calphas``
     - -
     - -
     - List of finer calphas to look for in cloud_dict

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - processedclist of all triggers in the cloud with calphas in finer_calphas

Docstring
---------

.. code-block:: text

   :param cloud_dict: Dictionary with processedclists indexed by template id
   :param finer_calphas: List of finer calphas to look for in cloud_dict
   :return:
       processedclist of all triggers in the cloud with calphas in
       finer_calphas
