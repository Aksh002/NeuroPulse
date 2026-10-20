import {describe,it,expect} from 'vitest';
import {formatMetric} from '../lib/api';
describe('unavailable predictions',()=>{it('does not turn null or nonfinite values into zero',()=>{
  expect(formatMetric(null)).toBe('Unavailable');expect(formatMetric(NaN)).toBe('Unavailable');
  expect(formatMetric(0)).toBe('0.0');expect(formatMetric(42.18)).toBe('42.2');
});});
