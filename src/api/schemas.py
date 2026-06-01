"""
Pydantic Validation Schemas
===========================
Defines the structure and validation contracts for incoming single/batch records
and outgoing threat classification responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class ConnectionRecord(BaseModel):
    """
    Validation schema for a single raw NSL-KDD network connection payload.
    Omits 'class' and 'difficulty_score' target/metadata columns.
    """
    duration: int = Field(..., description="Connection duration in seconds")
    protocol_type: str = Field(..., description="Protocol type (e.g. tcp, udp, icmp)")
    service: str = Field(..., description="Network service (e.g. http, private, smtp)")
    flag: str = Field(..., description="Status flag of the connection (e.g. SF, S0, REJ)")
    src_bytes: int = Field(..., description="Bytes sent from source to destination")
    dst_bytes: int = Field(..., description="Bytes sent from destination to source")
    land: int = Field(..., description="1 if source and destination IP/port are equal; 0 otherwise")
    wrong_fragment: int = Field(..., description="Number of wrong fragments")
    urgent: int = Field(..., description="Number of urgent packets")
    hot: int = Field(..., description="Number of 'hot' indicators")
    num_failed_logins: int = Field(..., description="Number of failed login attempts")
    logged_in: int = Field(..., description="1 if successfully logged in; 0 otherwise")
    num_compromised: int = Field(..., description="Number of compromised conditions")
    root_shell: int = Field(..., description="1 if root shell is obtained; 0 otherwise")
    su_attempted: int = Field(..., description="1 if 'su root' command attempted; 0 otherwise")
    num_root: int = Field(..., description="Number of 'root' accesses")
    num_file_creations: int = Field(..., description="Number of file creation operations")
    num_shells: int = Field(..., description="Number of shell prompts")
    num_access_files: int = Field(..., description="Number of operations on access control files")
    num_outbound_cmds: int = Field(..., description="Number of outbound commands in an ftp session")
    is_hot_login: int = Field(..., description="1 if login is hot; 0 otherwise")
    is_guest_login: int = Field(..., description="1 if login is guest; 0 otherwise")
    count: int = Field(..., description="Number of connections to the same host in the past two seconds")
    srv_count: int = Field(..., description="Number of connections to the same service in the past two seconds")
    serror_rate: float = Field(..., description="Percentage of connections that have 'SYN' errors")
    srv_serror_rate: float = Field(..., description="Percentage of connections that have 'SYN' errors to the same service")
    rerror_rate: float = Field(..., description="Percentage of connections that have 'REJ' errors")
    srv_rerror_rate: float = Field(..., description="Percentage of connections that have 'REJ' errors to the same service")
    same_srv_rate: float = Field(..., description="Percentage of connections to the same service")
    diff_srv_rate: float = Field(..., description="Percentage of connections to different services")
    srv_diff_host_rate: float = Field(..., description="Percentage of connections to different hosts to the same service")
    dst_host_count: int = Field(..., description="Destination host same count")
    dst_host_srv_count: int = Field(..., description="Destination host same service count")
    dst_host_same_srv_rate: float = Field(..., description="Destination host same service rate")
    dst_host_diff_srv_rate: float = Field(..., description="Destination host diff service rate")
    dst_host_same_src_port_rate: float = Field(..., description="Destination host same src port rate")
    dst_host_srv_diff_host_rate: float = Field(..., description="Destination host srv diff host rate")
    dst_host_serror_rate: float = Field(..., description="Destination host SYN error rate")
    dst_host_srv_serror_rate: float = Field(..., description="Destination host srv SYN error rate")
    dst_host_rerror_rate: float = Field(..., description="Destination host REJ error rate")
    dst_host_srv_rerror_rate: float = Field(..., description="Destination host srv REJ error rate")

    model_config = {
        "json_schema_extra": {
            "example": {
                "duration": 0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 215,
                "dst_bytes": 4507,
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": 0,
                "num_failed_logins": 0,
                "logged_in": 1,
                "num_compromised": 0,
                "root_shell": 0,
                "su_attempted": 0,
                "num_root": 0,
                "num_file_creations": 0,
                "num_shells": 0,
                "num_access_files": 0,
                "num_outbound_cmds": 0,
                "is_hot_login": 0,
                "is_guest_login": 0,
                "count": 1,
                "srv_count": 1,
                "serror_rate": 0.0,
                "srv_serror_rate": 0.0,
                "rerror_rate": 0.0,
                "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": 51,
                "dst_host_srv_count": 255,
                "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.02,
                "dst_host_srv_diff_host_rate": 0.05,
                "dst_host_serror_rate": 0.0,
                "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_rerror_rate": 0.0
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Serving output response detailing Zero-Day hybrid prediction classification.
    """
    is_attack: bool = Field(..., description="True if connection represents an intrusion attempt")
    attack_family: str = Field(..., description="Target family name (e.g. normal, dos, probe, r2l, u2r)")
    confidence: float = Field(..., description="Pipeline probability scoring (0.0 to 1.0)")
    stage1_score: float = Field(..., description="Stage 1 unsupervised anomaly score (Isolation Forest shifted score)")


class BatchPredictionResponse(BaseModel):
    """
    Structured response array for multi-record batch queries.
    """
    predictions: List[PredictionResponse] = Field(..., description="Array of prediction outcomes")
    count: int = Field(..., description="Total connections successfully parsed and classified")
