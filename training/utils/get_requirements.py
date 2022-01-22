"""
 File name   : get_requarements.py
 Description : description

 Date created : 15.10.2021
 Author:  Ihar Khakholka
"""

import pkg_resources


def get_requirements() -> str:
    """ Get Python virtual environment modules in text format:
        module1==version1
        module2==version2
        .....
    """
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                      for i in installed_packages])
    return '\n'.join(installed_packages_list)