'use strict';

/*!
 * TrustMark Node.js SDK
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 */

const { createTrustmark, capabilities } = require('./lib/trustmark');
const { TrustmarkError } = require('./lib/errors');
const { encodeVerified, WatermarkVerificationError } = require('./lib/verified');

module.exports = { createTrustmark, capabilities, encodeVerified, TrustmarkError, WatermarkVerificationError };
