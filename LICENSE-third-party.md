# Third-Party Software Notices

OpenAgent ships its own self-contained LLM execution runtime under
`src/core/_runner/`, `src/core/_run_state/`, `src/memory/sessions/`,
`src/memory/store/`, `src/models/providers/`, `src/mcp/_runtime/`, and
`src/stream/media.py`. The implementation of this runtime was originally
derived from the **Agno** framework (https://github.com/agno-agi/agno),
which is distributed under the **Apache License 2.0**.

The code has since been substantially restructured, renamed, and trimmed
for use in OpenAgent — unused subsystems (knowledge bases, vector DBs,
reasoning chains, evals, compression, culture/i18n, skills, the
Agno-internal workflow engine and scheduler, etc.) were removed; class
and module names were rewritten to match OpenAgent's package structure;
identifier names were brought in line with OpenAgent's naming conventions.

In accordance with the upstream license, the following acknowledgment is
preserved here:

> Copyright (c) Agno AGI Inc. and contributors.
> Licensed under the Apache License 2.0.
> The original source is available at https://github.com/agno-agi/agno.

A copy of the Apache License 2.0 follows.

---

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

(Full Apache 2.0 license text: https://www.apache.org/licenses/LICENSE-2.0.txt)
