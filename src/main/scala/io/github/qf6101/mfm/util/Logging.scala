package io.github.qf6101.mfm.util

import org.apache.log4j.Logger

/**
  * Created by qfeng on 16-8-30.
  */
trait Logging {
  private[this] val LOGGER = Logger.getLogger(this.getClass.toString)

  protected def logInfo(msg: String, t: Throwable = null): Unit = {
    if (t == null) LOGGER.info(msg) else LOGGER.info(msg, t)
  }

  protected def logError(msg: String, t: Throwable = null): Unit = {
    if (t == null) LOGGER.error(msg) else LOGGER.error(msg, t)
  }

  protected def logWarning(msg: String, t: Throwable = null): Unit = {
    if (t == null) LOGGER.warn(msg) else LOGGER.warn(msg, t)
  }

  protected def logDebug(msg: String, t: Throwable = null): Unit = {
    if (t == null) LOGGER.debug(msg) else LOGGER.debug(msg, t)
  }
}
