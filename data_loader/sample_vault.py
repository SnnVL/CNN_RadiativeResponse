"""Sample vault modules for storing data samples in dictionaries.

Classes
---------
SampleDict(dict)

"""

import numpy as np
import copy


class SampleDict(dict):
    def __init__(self, *arg, **kw):
        super(SampleDict, self).__init__(*arg, **kw)

        self.__setitem__("x", [])
        self.__setitem__("y", [])
        self.__setitem__("year", [])
        self.__setitem__("member", [])
        self.__setitem__("model", [])

    def summary(self):
        for key in self:
            print(f"data[{key}].shape = {self[key].shape}")
        print("\n")

    def reset(self):
        for key in self:
            self[key] = []

    def reshape(self):
        for key in self:
            if len(self[key]) == 0:
                continue

            if self[key].ndim == 5:
                self[key] = self[key].reshape(
                    (
                        self[key].shape[0] * self[key].shape[1],
                        self[key].shape[2],
                        self[key].shape[3],
                        self[key].shape[4],
                    )
                )
            elif self[key].ndim == 2:
                self[key] = self[key].reshape(
                    (
                        self[key].shape[0] * self[key].shape[1], 
                        1,
                    )
                )
            else:
                raise NotImplementedError

    def subsample(self, idx, axis=0, use_copy=False):
        if use_copy:
            d = copy.deepcopy(self)
            if axis == 0:
                for key in d:
                    d[key] = d[key][idx, ...]
            elif axis == 1:
                for key in d:
                    d[key] = d[key][:, idx, ...]
            else:
                raise NotImplementedError
            return d

        elif not use_copy:
            if axis == 0:
                for key in self:
                    self[key] = self[key][idx, ...]
            elif axis == 1:
                for key in self:
                    self[key] = self[key][:, idx, ...]
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def concat(self, f_dict):
        for key in self:
            if len(self[key]) == 0:
                self[key] = f_dict[key]
            elif len(f_dict[key]) == 0:
                pass
            else:
                self[key] = np.concatenate((self[key], f_dict[key]), axis=0)
