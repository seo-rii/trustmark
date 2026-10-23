'use strict';

const { TrustmarkError, throwIfAborted } = require('./errors');

/**
 * FIFO job queue that runs at most `concurrency` jobs at a time and holds at most `maxQueue`
 * waiting jobs. A waiting job whose signal aborts leaves the queue without running.
 */
class JobQueue {
  constructor(concurrency, maxQueue) {
    this.concurrency = concurrency;
    this.maxQueue = maxQueue;
    this.active = 0;
    this.waiting = [];
    this.idle = [];
    this.closed = false;
  }

  /** Runs `job()` when a slot is free; rejects with the abort reason if `signal` aborts first. */
  run(job, signal) {
    if (this.closed) return Promise.reject(new TrustmarkError('CLOSED', 'The TrustMark instance is closed'));
    try {
      throwIfAborted(signal);
    } catch (error) {
      return Promise.reject(error);
    }
    if (this.active < this.concurrency) return this.start(job);
    if (this.waiting.length >= this.maxQueue) {
      return Promise.reject(new TrustmarkError('QUEUE_FULL', `${this.waiting.length} jobs are already waiting (maxQueue: ${this.maxQueue})`));
    }
    return new Promise((resolve, reject) => {
      const entry = { job, resolve, reject, signal, onAbort: null };
      if (signal) {
        entry.onAbort = () => {
          const index = this.waiting.indexOf(entry);
          if (index >= 0) this.waiting.splice(index, 1);
          try { throwIfAborted(signal); } catch (error) { reject(error); }
        };
        signal.addEventListener('abort', entry.onAbort, { once: true });
      }
      this.waiting.push(entry);
    });
  }

  start(job) {
    this.active++;
    let result;
    try {
      result = Promise.resolve(job());
    } catch (error) {
      result = Promise.reject(error);
    }
    const done = () => {
      this.active--;
      this.next();
    };
    result.then(done, done);
    return result;
  }

  next() {
    while (this.active < this.concurrency && this.waiting.length > 0) {
      const entry = this.waiting.shift();
      if (entry.signal) entry.signal.removeEventListener('abort', entry.onAbort);
      this.start(entry.job).then(entry.resolve, entry.reject);
    }
    if (this.active === 0 && this.waiting.length === 0) {
      for (const resolve of this.idle.splice(0)) resolve();
    }
  }

  /** Rejects the waiting jobs with CLOSED and resolves once the running jobs have settled. */
  close() {
    this.closed = true;
    for (const entry of this.waiting.splice(0)) {
      if (entry.signal) entry.signal.removeEventListener('abort', entry.onAbort);
      entry.reject(new TrustmarkError('CLOSED', 'The TrustMark instance was closed before this job started'));
    }
    if (this.active === 0) return Promise.resolve();
    return new Promise((resolve) => this.idle.push(resolve));
  }
}

module.exports = { JobQueue };
