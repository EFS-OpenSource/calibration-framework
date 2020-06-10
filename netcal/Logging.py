# Copyright (C) 2019-2020 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND Visteon Electronics Germany GmbH, Kerpen, Germany
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import logging
import io


class TqdmHandler(io.StringIO):
    """
    Output tqdm messages via logging module.
    """

    logger = None
    level = None
    buf = ''

    def __init__(self, logger: logging.Logger, level: int = None):
        """
        Constructor. Initialize logging stream handler.
        """

        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf: str):
        """
        Write logging message into buffer

        Parameters
        ----------
        buf : str
            Message that shall be logged.
        """

        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        """
        Flush buffered messages to logger
        """

        self.logger.log(self.level, self.buf)
