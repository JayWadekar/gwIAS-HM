utils.extract_parts
===================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Extracts the parts of the directory name based on the prefix and suffix

Signature
---------

.. code-block:: python

   def extract_parts(directory_name, prefix, suffix)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``directory_name``
     - -
     - -
     - Subdirectory name to extract parts from
   * - ``prefix``
     - -
     - -
     - Prefix to match the subdirectories for the given run and source type, can have wildcards (see BBH_PREFIXES etc., for examples). The prefix should match the part of the subdirectory name before the chirp mass id if the chirp mass id is absent, it makes it zero (like O1 BNS) The part after the chirp mass id is supposed to end with the subbank id
   * - ``suffix``
     - -
     - -
     - Any desired suffix to demand at the end of the subdirectory's name (cand type is the use case). Can be the empty string

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

   Extracts the parts of the directory name based on the prefix and suffix
   :param directory_name: Subdirectory name to extract parts from
   :param prefix:
       Prefix to match the subdirectories for the given run and source type,
       can have wildcards (see BBH_PREFIXES etc., for examples). The prefix
       should match the part of the subdirectory name before the chirp mass id
       if the chirp mass id is absent, it makes it zero (like O1 BNS)
       The part after the chirp mass id is supposed to end with the subbank id
   :param suffix:
       Any desired suffix to demand at the end of the subdirectory's name
       (cand type is the use case). Can be the empty string
   :return:
