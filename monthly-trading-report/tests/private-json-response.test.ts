import assert from "node:assert/strict";
import test from "node:test";
import { privateJsonResponse } from "../lib/private-json-response";

test("private revalidation returns full JSON, bodyless 304, and changed JSON",async()=>{
  const first=privateJsonResponse(new Request("http://localhost"),{trades:[1]});
  assert.equal(first.status,200); assert.match(first.headers.get("cache-control")!,/private, no-cache/);
  assert.equal(first.headers.get("vary"),"Cookie");
  const etag=first.headers.get("etag")!;
  assert.deepEqual(await first.json(),{trades:[1]});
  for(const value of [etag,etag.replace('W/',''),`"other", ${etag}`,"*"]) {
    const response=privateJsonResponse(new Request("http://localhost",{headers:{"if-none-match":value}}),{trades:[1]});
    assert.equal(response.status,304);assert.equal(await response.text(),"");
  }
  const changed=privateJsonResponse(new Request("http://localhost",{headers:{"if-none-match":etag}}),{trades:[2]});
  assert.equal(changed.status,200);assert.deepEqual(await changed.json(),{trades:[2]});
});
