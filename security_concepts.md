# Security concepts — attack types and terminology

Grounded in the actual NGIDS-DS ground truth where possible (queried directly from
`dataset/raw_backup/host_logs.parquet`), not generic textbook definitions pasted in
blind. Where a definition is standard security terminology rather than something
specific to this dataset, that's noted.

---

## Core terminology (say these correctly, cold)

- **Vulnerability** — a flaw in software (a bug, a design mistake, a missing check)
  that could be abused to make the software do something its designers didn't intend.
  A vulnerability existing doesn't mean anyone has used it yet.
- **Exploit** — the actual code or technique that *triggers* a vulnerability to achieve
  a specific effect (crash the program, leak memory, execute attacker-controlled code).
  The vulnerability is the weakness; the exploit is the weapon built to use it.
- **Payload** — what the exploit actually does once it has control — e.g. open a
  network shell, drop a file, escalate privileges. Distinct from the exploit itself,
  which is just the delivery mechanism to reach the point of executing the payload.
- **Shellcode** — a specific, usually small, form of payload: raw machine code
  (traditionally code that spawns a command shell, hence the name) injected directly
  into a process's memory and executed as the result of a successful exploit. Not every
  payload is shellcode, but shellcode is the classic payload in memory-corruption
  exploits specifically.
- **CVE (Common Vulnerabilities and Exposures)** — a public, standardised catalogue
  identifier for a specific known vulnerability, in the format `CVE-YYYY-NNNNN`. It's a
  name, not a severity rating (CVSS is the separate severity scoring system). Two real
  examples that appear directly in this dataset's own ground truth:
  `CVE-2013-1313` (Microsoft Office OLE Automation Integer Overflow) and
  `CVE-2010-0018` (Microsoft Embedded OpenType Font LZCOMP Decompressor Array Index
  Overflow) — both genuine, both cited with their CVE reference and a description in
  `ground_truth.parquet`.
- **Zero-day** — a vulnerability being actively exploited before the vendor has issued
  (or sometimes even knows about) a fix. None of NGIDS-DS's attacks are zero-days in
  this sense — they're all replays of previously-disclosed, CVE-numbered vulnerabilities
  against a deliberately unpatched test host.
- **IDS vs. IPS** — an Intrusion *Detection* System observes and alerts; an Intrusion
  *Prevention* System sits inline and can actively block traffic/activity. This project
  builds a detector (a classifier producing an alert), not a prevention system — it
  never acts on its own output.
- **Signature-based vs. anomaly-based detection** — signature-based matches activity
  against a database of known-bad patterns (fast, precise, but blind to anything novel);
  anomaly-based learns a model of "normal" and flags deviations from it (can catch novel
  attacks, but is more prone to both false positives on unusual-but-benign activity and
  false negatives on attacks that resemble normal behaviour). This project is
  anomaly-based, though technically supervised (trained on labelled attack examples,
  not purely unsupervised deviation detection).
- **False positive / false negative, in this context** — a false positive is a normal
  trace incorrectly flagged as an attack (costs analyst time investigating a non-issue);
  a false negative is a real attack the detector misses entirely (the more dangerous
  failure mode, which is exactly why the F2 threshold in this study is tuned to favour
  recall over precision).
- **Syscall (system call)** — the interface a running process uses to ask the operating
  system kernel to do something on its behalf — open a file, read from the network,
  create a child process, allocate memory. Every meaningful thing a process does that
  touches the outside world or system resources goes through a syscall, which is
  exactly why syscall sequences are a rich, hard-to-fake signal for host-based
  detection: an attacker can hide their *intent* in application-level logs, but can't
  avoid making the underlying kernel calls needed to actually act.

---

## How NGIDS-DS's attacks were actually generated

Worth knowing if asked "were these simulated or real attacks": the CVE-referenced
`attack_refrence` entries in the dataset's ground truth point to URLs on
`strikecenter.bpointsys.com` — this is BreakingPoint Systems (BPS), a commercial
network/security test-traffic generation platform (later acquired by Ixia). The
`Generic` category's own subcategory is literally labelled `"IXIA Batch"` in the host
logs. So the attacks are **real, working exploit payloads for genuine CVEs**, replayed
against a live, deliberately vulnerable Linux host using a commercial attack-simulation
appliance — not toy or synthetic attack signatures. That's a meaningfully stronger claim
than "the labels say attack" alone.

---

## The seven attack categories

For each: the general security definition, then how it shows up concretely in this
dataset (real subcategory names and event counts, queried directly from
`host_logs.parquet` — not the messier `ground_truth.parquet`, which has structural
inconsistencies across its rows and shouldn't be quoted for exact per-attack detail;
see `viva_qna.md` B8).

### Exploits (900,828 events — the largest category)

**General definition:** an attack that directly triggers a specific software
vulnerability to gain some effect the software wasn't supposed to allow — typically
code execution, privilege escalation, or an information leak.

**In this dataset:** the largest and most varied category by far, dominated by
office-document and browser-delivered payloads: `Office Document Batch` (276,578
events), `Browser` (152,319 + 17,271 events across two logging passes), `Clientside`
(102,893), `Clientside Microsoft Office Batch`, `Clientside Microsoft Paint`, `Browser
FTP Batch`, plus a long tail of protocol-specific exploits (`SMB Batch`, `Microsoft IIS
Batch`, `DCERPC Batch`, `Web Application Cross-Site Scripting Batch`, `LDAP`, `ICMP`,
`RDesktop`, `WINS`, `TFTP`, `DNS`, `POP3`, `PPTP`, `MSSQL`, `NNTP`). This is the category
your worked example (the Firefox trace) belongs to — subcategory `Browser`.

### Denial of Service (129,185 events)

**General definition:** an attack whose goal is disrupting availability — making a
service, process, or resource unusable for legitimate users — rather than gaining
unauthorised access or code execution. Can work by resource exhaustion (flooding) or by
triggering a crash/hang via a targeted bug.

**In this dataset:** protocol- and service-targeted floods/crashes: `Microsoft Office
Batch` (38,646), `Browser Batch` (26,935), `NetBIOS/SMB Batch` (16,823), `IIS Web
Server` (12,277), `Windows Explorer`, `HTTP`, `Hypervisor`, `TCP`, `FTP`, `SMTP`, `SNMP`,
`TFTP`, `DCERPC`, `LDAP` — each subcategory naming the specific protocol/service being
targeted for disruption.

### Generic (79,624 events)

**General definition (standard usage in the UNSW-NB15 taxonomy this dataset's category
scheme descends from, via Haider's thesis):** a catch-all for technique-agnostic
attacks — historically defined in that taxonomy as attacks against block ciphers using
a hash function to force a collision, independent of the cipher's specific structure.

**Honest caveat:** the only subcategory actually present in this dataset's host logs for
`Generic` is `"IXIA Batch"` (79,624 events, all under that one label) — there isn't
enough subcategory detail in the data itself to confirm the textbook cryptographic
definition is precisely what's being exercised here versus a broader "miscellaneous
batch from the attack tool" grouping. If asked to define this category, give the
textbook definition but don't claim you've verified it against this dataset's specific
payloads — you haven't, and the data doesn't give you enough to.

### Backdoors (70,712 events)

**General definition:** a mechanism — often installed after an initial compromise —
that bypasses normal authentication or access control to give an attacker persistent,
repeatable access to a system without needing to re-exploit the original vulnerability
each time.

**In this dataset:** a single subcategory, `"All Batch"` (70,712 events, verified
directly against `host_logs.parquet`) — less granular in the logs than Exploits or DoS.
*Trace count deliberately not stated here* — an earlier version of this file claimed
"~61 traces," which does not match either of the two natural ways to count it (173
traces contain at least one Backdoors-labelled event; 14 traces are Backdoors-only with
no other category present), so that figure was unverifiable and has been removed. If
asked for a trace count live, say you'd need to check rather than quote a number from
memory.

### Shellcode (57,263 events)

**General definition:** see the glossary above — raw injected machine code executed as
an exploit's payload, classically to spawn a command shell for the attacker.

**In this dataset:** `Linux Batch` (44,245 events) and `Multiple OS Batch`
(13,018 events) — i.e. shellcode payloads targeting Linux specifically, and a separate
batch of cross-platform shellcode payloads. Event counts verified against
`host_logs.parquet`; the trace counts an earlier version of this file gave ("~56", "~26")
did not check out against direct verification and have been removed for the same reason
as Backdoors above.

### Worms (14,125 events)

**General definition:** self-replicating malware that spreads autonomously across
hosts/networks, typically by exploiting a vulnerability to copy itself onto new
targets, without needing to attach to a host file or wait for a user to run something
(the key distinction from a traditional virus, which needs a host program/file and
usually user action to spread).

**In this dataset:** one subcategory, `"All Batch"` (14,125 events, verified). The "19
traces" figure an earlier version of this file gave did not check out (direct count:
90 traces contain at least one Worms-labelled event, 0 are Worms-only) and has been
removed.

### Reconnaissance (10,690 events — smallest by event count)

**General definition:** pre-attack information gathering — port scanning, service
enumeration, OS/software fingerprinting — used to map a target's attack surface before
an actual exploit is launched. Often the quietest, most normal-looking category, since
successful reconnaissance is deliberately designed not to trigger obvious defences.

**In this dataset:** `FrontPage HTTP Batch` (9,515 + 917 events) and `Microsoft IIS HTTP
200/200+A308969` (258 events) — HTTP-based probing against FrontPage and IIS web
servers, consistent with the general definition of enumeration/fingerprinting activity.

---

## Likely security-fundamentals viva questions

### "What's the difference between a vulnerability, an exploit, and a payload?"

**Say:** A vulnerability is the underlying flaw. An exploit is the mechanism that
triggers it. A payload is what runs once the exploit has succeeded. In this dataset,
every attack event ultimately traces back to a specific CVE-numbered vulnerability being
exploited by a real, working exploit built for that CVE.

### "Why syscalls specifically, rather than network traffic or application logs?"

**Say:** Syscalls sit at the boundary between a process and the operating system, so
they capture what a process actually *did* on the host — opened a file, spawned a
child, read from a socket — regardless of what the attack looked like at the network or
application-log level. An attacker can often blend in at the network layer or suppress
application logging, but can't act on the host at all without going through the kernel
interface. That's what makes host-based, syscall-level detection a meaningfully
different signal from network-flow-based detection (the kind most Related Work papers
in this study actually use).

### "Is this anomaly detection or misuse/signature detection, really? You trained on labelled attacks."

**Say:** Fair distinction to draw precisely. It's supervised learning trained on
labelled examples of both classes, which technically makes it closer to a learned
misuse detector than a purely unsupervised "model normal, flag deviations" anomaly
detector. It's called anomaly-based in the framing sense used throughout this
literature (deep-learning IDS trained on behavioural features rather than static
signatures), but if pushed on the precise taxonomy, the honest answer is: supervised
learning on behavioural features, not unsupervised outlier detection, and not classic
hash/pattern signature matching either.

### "Which attack categories would this detector actually be useful against in the real world?"

**Say:** The ones that leave a real host-level footprint — Exploits and DoS both
directly execute or disrupt something on the host, which is exactly where syscall-level
detection has the most to work with. Reconnaissance is deliberately designed to look
unremarkable, which is a harder detection target for any host-based approach, not just
this one — this project doesn't report a per-category breakdown in the final paper (see
`viva_qna.md` B6 for why), so don't quote a specific per-category recall number live;
speak to it qualitatively as above.

### "What does 'hands-on-keyboard' mean, and how does it relate to any of these seven categories?"

**Say:** It refers to an attacker manually interacting with a compromised system in
real time — running commands, exploring the environment — rather than relying on
pre-packaged malware doing everything autonomously. None of NGIDS-DS's seven categories
are specifically "hands-on-keyboard" activity; they're all automated exploit/attack-tool
traffic replayed by BreakingPoint. It's worth being able to name that distinction
plainly if an examiner connects it back to the CrowdStrike statistic in the
introduction — the motivating statistic is about a broader industry trend, not a claim
that this specific dataset contains hands-on-keyboard sessions.
