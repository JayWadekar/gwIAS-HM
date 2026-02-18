data_operations.makeasdplots
============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Makes comparison plots of ASDs with the data-derived one

Signature
---------

.. code-block:: python

   def makeasdplots(asddir, asd_function = None, show = True, fname = 'ASDS.pdf')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``asddir``
     - -
     - -
     - Directory with ASD files
   * - ``asd_function``
     - -
     - None
     - Function that takes frequencies (in Hz) and returns data-derived asd (in 1/Hz\\\*\\\*0.5)
   * - ``show``
     - -
     - True
     - Flag indicated whether to show or save
   * - ``fname``
     - -
     - 'ASDS.pdf'
     - Filename

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

   Makes comparison plots of ASDs with the data-derived one
   :param asddir: Directory with ASD files
   :param asd_function: Function that takes frequencies (in Hz) and returns
                    data-derived asd (in 1/Hz**0.5)
   :param show: Flag indicated whether to show or save
   :param fname: Filename
   :return:
