"""
Pydantic models for request and response validation.
Used throughout the FastAPI application for type safety and API documentation.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# REQUEST MODELS
# -----------------------------------------------------------------------------

class ScanPathRequest(BaseModel):
    """Request model for scanning a directory path."""
    root: str = Field(..., description="Root directory path to scan for photos")


class TagItem(BaseModel):
    """Tag item - can be a string or dict with tag and score."""
    tag: str = Field(..., description="Tag name")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Tag confidence score")


class AddPhotoTagsRequest(BaseModel):
    """Request model for adding tags to a photo."""
    tags: List[str | TagItem] = Field(..., description="List of tags (strings or TagItem objects)")


class IndexEmbeddingsRequest(BaseModel):
    """Request model for indexing embeddings."""
    incremental: bool = Field(default=True, description="Only index photos without embeddings")


class IndexFacesRequest(BaseModel):
    """Request model for indexing faces."""
    limit: Optional[int] = Field(None, ge=1, description="Maximum number of photos to process")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum face detection score")
    thumb_size: int = Field(default=160, ge=32, le=512, description="Thumbnail size in pixels")


class ClusterFacesRequest(BaseModel):
    """Request model for clustering faces."""
    threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity threshold for clustering")
    reset: bool = Field(default=False, description="If True, unassign all faces before clustering (re-cluster everything)")


class CreatePersonRequest(BaseModel):
    """Request model for creating a person."""
    name: str = Field(..., min_length=1, description="Person name")


class UpdatePersonRequest(BaseModel):
    """Request model for updating a person."""
    name: str = Field(..., min_length=1, description="New person name")


class AssignFacePersonRequest(BaseModel):
    """Request model for assigning a person to a face."""
    person_id: int = Field(..., ge=1, description="Person ID to assign")


class MergePersonsRequest(BaseModel):
    """Request model for merging persons."""
    source_person_id: int = Field(..., ge=1, description="Source person ID to merge from")


# -----------------------------------------------------------------------------
# RESPONSE MODELS
# -----------------------------------------------------------------------------

class PhotoResponse(BaseModel):
    """Photo response model."""
    id: int = Field(..., description="Photo ID")
    file_path: str = Field(..., description="File path in storage")
    size: Optional[int] = Field(None, description="File size in bytes")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    checksum: Optional[str] = Field(None, description="File checksum")
    phash: Optional[str] = Field(None, description="Perceptual hash")
    exif_datetime: Optional[str] = Field(None, description="EXIF datetime")
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")
    thumb_path: Optional[str] = Field(None, description="Thumbnail path in storage")
    score: Optional[float] = Field(None, description="Search relevance score (for search results)")

    class Config:
        from_attributes = True


class PhotoListResponse(BaseModel):
    """Response model for photo list."""
    items: List[PhotoResponse] = Field(..., description="List of photos")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    total: int = Field(..., description="Total number of photos")


class TagResponse(BaseModel):
    """Tag response model."""
    tag: str = Field(..., description="Tag name")
    score: float = Field(..., description="Tag confidence score")


class TagListResponse(BaseModel):
    """Response model for tag list."""
    photo_id: int = Field(..., description="Photo ID")
    tags: List[TagResponse] = Field(..., description="List of tags")


class FaceResponse(BaseModel):
    """Face response model."""
    face_id: int = Field(..., description="Face ID")
    photo_id: int = Field(..., description="Photo ID")
    photo_path: Optional[str] = Field(None, description="Photo file path")
    thumb_path: Optional[str] = Field(None, description="Face thumbnail path")
    person_id: Optional[int] = Field(None, description="Assigned person ID")
    person_name: Optional[str] = Field(None, description="Assigned person name")


class FaceListResponse(BaseModel):
    """Response model for face list."""
    items: List[FaceResponse] = Field(..., description="List of faces")


class PersonResponse(BaseModel):
    """Person response model."""
    id: int = Field(..., description="Person ID")
    name: str = Field(..., description="Person name")


class PersonListResponse(BaseModel):
    """Response model for person list."""
    persons: List[PersonResponse] = Field(..., description="List of persons")


class PersonFacesResponse(BaseModel):
    """Response model for person's faces."""
    items: List[Dict[str, Any]] = Field(..., description="List of face thumbnails for the person")


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    created_at: Optional[str] = Field(None, description="Job creation timestamp")
    started_at: Optional[str] = Field(None, description="Job start timestamp")
    ended_at: Optional[str] = Field(None, description="Job end timestamp")
    result: Optional[Any] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Job error message")


class IndexJobResponse(BaseModel):
    """Response model for indexing job submission."""
    indexed: int = Field(default=0, description="Number of items indexed")
    job_id: Optional[str] = Field(None, description="Job ID")
    message: str = Field(..., description="Status message")
    status: Optional[str] = Field(None, description="Job status")


class FaceIndexJobResponse(BaseModel):
    """Response model for face indexing job submission."""
    processed_photos: int = Field(default=0, description="Number of photos processed")
    job_id: Optional[str] = Field(None, description="Job ID")
    message: str = Field(..., description="Status message")
    status: Optional[str] = Field(None, description="Job status")


class ClusterFacesResponse(BaseModel):
    """Response model for face clustering."""
    clustered_faces: int = Field(..., description="Number of faces clustered")
    persons_created: int = Field(..., description="Number of persons created")
    threshold: float = Field(..., description="Similarity threshold used")
    job_id: Optional[str] = Field(None, description="Background job ID if clustering was queued")
    message: Optional[str] = Field(None, description="Status message")


class SearchResultsResponse(BaseModel):
    """Response model for search results."""
    query: str = Field(..., description="Search query")
    results: List[PhotoResponse] = Field(..., description="Search results")


class ScanPathResponse(BaseModel):
    """Response model for scan path operation."""
    root: str = Field(..., description="Scanned root path")
    processed: int = Field(..., description="Number of photos processed")


class UploadPhotoResponse(BaseModel):
    """Response model for single photo upload."""
    id: int = Field(..., description="Photo ID")
    file_path: str = Field(..., description="File path in storage")
    storage_url: str = Field(..., description="Storage URL")
    thumbnail: str = Field(..., description="Thumbnail path")
    thumbnail_url: str = Field(..., description="Thumbnail URL")


class UploadPhotosResponse(BaseModel):
    """Response model for photo upload (multiple files)."""
    saved: List[UploadPhotoResponse | Dict[str, Any]] = Field(..., description="Successfully uploaded photos")
    errors: Optional[List[str]] = Field(None, description="Upload errors")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    db_path: str = Field(..., description="Database path")
    db_status: Optional[str] = Field(None, description="Database status")
    storage_status: Optional[str] = Field(None, description="Storage status")
    storage_endpoint: Optional[str] = Field(None, description="Storage endpoint")
    storage_bucket: Optional[str] = Field(None, description="Storage bucket name")
    storage_note: Optional[str] = Field(None, description="Storage note")


class RootResponse(BaseModel):
    """Response model for root endpoint."""
    message: str = Field(..., description="API message")
    version: str = Field(..., description="API version")
    docs: str = Field(..., description="API documentation URL")
    health: str = Field(..., description="Health check URL")


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")


# -----------------------------------------------------------------------------
# AUTHENTICATION SCHEMAS
# -----------------------------------------------------------------------------

class UserCreateRequest(BaseModel):
    """Request model for user registration."""
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    email: Optional[str] = Field(None, description="Email address (optional)")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User response model."""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    role: str = Field(..., description="User role (admin or user)")
    is_active: bool = Field(..., description="Whether account is active")
    is_whitelisted: bool = Field(..., description="Whether user is whitelisted")
    whitelisted_at: Optional[str] = Field(None, description="When user was whitelisted")
    created_at: Optional[str] = Field(None, description="Account creation timestamp")
    last_login: Optional[str] = Field(None, description="Last login timestamp")

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User information")


class RegisterResponse(BaseModel):
    """Response model for user registration."""
    message: str = Field(..., description="Registration status message")
    username: str = Field(..., description="Registered username")
    status: str = Field(default="pending_approval", description="Account status")


class PendingUserResponse(BaseModel):
    """Response for pending approval users."""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    created_at: str = Field(..., description="Account creation timestamp")
    status: str = Field(default="pending_approval", description="Account status")


class UserListResponse(BaseModel):
    """Response for user list."""
    users: List[UserResponse] = Field(..., description="List of users")
    pending_count: int = Field(default=0, description="Count of pending users")


class WhitelistUserRequest(BaseModel):
    """Request to whitelist a user."""
    user_id: int = Field(..., description="User ID to whitelist")


class UpdateUserRoleRequest(BaseModel):
    """Request to update user role."""
    role: str = Field(..., description="New role (admin or user)")

